<script setup lang="ts">
import { createAiAvatarKitClient } from '@proj-airi/stage-ui/libs'
import { usePipelineCharacterSpeechPlaybackQueueStore } from '@proj-airi/stage-ui/composables/queues'
import { EMOTION_EmotionMotionName_value, Emotion } from '@proj-airi/stage-ui/constants/emotions'
import { useAudioContext } from '@proj-airi/stage-ui/stores/audio'
import { defaultModelParameters, useLive2d } from '@proj-airi/stage-ui/stores/live2d'
import { storeToRefs } from 'pinia'
import { useLocalStorage } from '@vueuse/core'
import { onMounted, onUnmounted, ref, watch } from 'vue'

const wsUrl = ref('ws://localhost:8000/ws')
const sessionId = ref('airi_session')
const userId = ref('airi_user')
const inputText = ref('')

const connected = ref(false)
const micEnabled = ref(false)
const screenEnabled = ref(false)
const cameraEnabled = ref(false)
const micLevel = ref(0)
const isPlaying = ref(false)
const logs = ref<string[]>([])

const screenIntervalMs = useLocalStorage('aiavatarkit-vision-screen-interval', 1000)
const cameraIntervalMs = useLocalStorage('aiavatarkit-vision-camera-interval', 1000)
const diffThreshold = useLocalStorage('aiavatarkit-vision-diff-threshold', 0.08)
const diffDownscaleWidth = useLocalStorage('aiavatarkit-vision-downscale-width', 64)
const diffDownscaleHeight = useLocalStorage('aiavatarkit-vision-downscale-height', 36)
const screenFrameRate = useLocalStorage('aiavatarkit-vision-screen-fps', 5)
const cameraFrameRate = useLocalStorage('aiavatarkit-vision-camera-fps', 5)

const screenHotkey = useLocalStorage('aiavatarkit-hotkey-screen', 'Ctrl+Shift+S')
const cameraHotkey = useLocalStorage('aiavatarkit-hotkey-camera', 'Ctrl+Shift+C')

const { audioContext } = useAudioContext()
const playbackStore = usePipelineCharacterSpeechPlaybackQueueStore()
const { connectAudioContext, connectAudioAnalyser, clearAll, onPlaybackStarted, onPlaybackFinished } = playbackStore
const { playbackQueue } = storeToRefs(playbackStore)

const live2dStore = useLive2d()
const { currentMotion, emotionMotionMap, modelParameters } = storeToRefs(live2dStore)

const audioAnalyser = ref<AnalyserNode>()

function log(line: string) {
  logs.value = [line, ...logs.value].slice(0, 80)
}

function isEditableTarget(target: EventTarget | null) {
  if (!(target instanceof HTMLElement))
    return false
  return target.tagName === 'INPUT'
    || target.tagName === 'TEXTAREA'
    || target.isContentEditable
}

function parseHotkey(hotkey: string) {
  const parts = hotkey.toLowerCase().split('+').map(part => part.trim()).filter(Boolean)
  const modifiers = new Set(parts.filter(part => ['ctrl', 'control', 'shift', 'alt', 'meta', 'cmd', 'command'].includes(part)))
  const key = parts.find(part => !modifiers.has(part)) || ''
  return {
    key,
    ctrl: modifiers.has('ctrl') || modifiers.has('control'),
    shift: modifiers.has('shift'),
    alt: modifiers.has('alt'),
    meta: modifiers.has('meta') || modifiers.has('cmd') || modifiers.has('command'),
  }
}

function matchesHotkey(event: KeyboardEvent, hotkey: string) {
  if (!hotkey)
    return false
  const parsed = parseHotkey(hotkey)
  const key = event.key.toLowerCase()
  const normalizedKey = key === ' ' ? 'space' : key
  return parsed.key === normalizedKey
    && event.ctrlKey === parsed.ctrl
    && event.shiftKey === parsed.shift
    && event.altKey === parsed.alt
    && event.metaKey === parsed.meta
}

type Live2DParameters = typeof defaultModelParameters

const FACE_TO_EMOTION: Record<string, Emotion> = {
  neutral: Emotion.Idle,
  joy: Emotion.Happy,
  happy: Emotion.Happy,
  angry: Emotion.Angry,
  sorrow: Emotion.Sad,
  sad: Emotion.Sad,
  fun: Emotion.Happy,
  surprised: Emotion.Surprise,
  surprise: Emotion.Surprise,
}

const FACE_PARAMETER_PRESETS: Record<string, Partial<Live2DParameters>> = {
  neutral: {},
  joy: {
    leftEyeOpen: 1,
    rightEyeOpen: 1,
    leftEyeSmile: 0.6,
    rightEyeSmile: 0.6,
    leftEyebrowY: 0.2,
    rightEyebrowY: 0.2,
    mouthOpen: 0.4,
    mouthForm: 0.6,
    cheek: 0.3,
  },
  angry: {
    leftEyeOpen: 0.6,
    rightEyeOpen: 0.6,
    leftEyebrowY: -0.3,
    rightEyebrowY: -0.3,
    leftEyebrowAngle: -0.4,
    rightEyebrowAngle: 0.4,
    mouthOpen: 0.2,
    mouthForm: -0.4,
  },
  sorrow: {
    leftEyeOpen: 0.4,
    rightEyeOpen: 0.4,
    leftEyebrowY: 0.3,
    rightEyebrowY: 0.3,
    leftEyebrowAngle: 0.3,
    rightEyebrowAngle: -0.3,
    mouthOpen: 0.1,
    mouthForm: -0.5,
  },
  fun: {
    leftEyeOpen: 0.8,
    rightEyeOpen: 0.8,
    leftEyeSmile: 0.8,
    rightEyeSmile: 0.8,
    mouthOpen: 0.5,
    mouthForm: 0.7,
    cheek: 0.35,
  },
  surprised: {
    leftEyeOpen: 1,
    rightEyeOpen: 1,
    leftEyebrowY: 0.5,
    rightEyebrowY: 0.5,
    mouthOpen: 0.8,
    mouthForm: 0.1,
  },
}

let faceResetTimer: number | null = null
let faceOverrideActive = false
let faceBaseParameters: Live2DParameters = { ...defaultModelParameters }

function applyFaceParameters(face: string, duration?: number) {
  const key = face.toLowerCase()
  const preset = FACE_PARAMETER_PRESETS[key]
  if (!preset)
    return

  if (faceResetTimer) {
    window.clearTimeout(faceResetTimer)
    faceResetTimer = null
  }

  if (!faceOverrideActive)
    faceBaseParameters = { ...modelParameters.value }

  if (key === 'neutral') {
    modelParameters.value = { ...faceBaseParameters }
    faceOverrideActive = false
    return
  }

  faceOverrideActive = true
  modelParameters.value = { ...faceBaseParameters, ...preset }

  if (duration && duration > 0) {
    faceResetTimer = window.setTimeout(() => {
      modelParameters.value = { ...faceBaseParameters }
      faceOverrideActive = false
      faceResetTimer = null
    }, duration * 1000)
  }
}

function applyEmotionMotion(face: string) {
  const emotion = FACE_TO_EMOTION[face.toLowerCase()]
  if (!emotion)
    return
  const mapped = emotionMotionMap.value[emotion]
  if (mapped)
    currentMotion.value = { group: mapped.group, index: mapped.index }
  else {
    const fallback = EMOTION_EmotionMotionName_value[emotion]
    if (fallback)
      currentMotion.value = { group: fallback }
  }
}

const client = createAiAvatarKitClient({
  audioContext,
  enqueueAudio: ({ audioBuffer, text }) => {
    playbackQueue.value.enqueue({ audioBuffer, text, special: null })
  },
  onText: (text) => {
    if (text)
      log(`ai: ${text}`)
  },
  onFace: (face, duration) => {
    log(`face: ${face}${duration ? ` (${duration}s)` : ''}`)
    applyFaceParameters(face, duration)
    applyEmotionMotion(face)
  },
  onMicLevel: (level) => {
    micLevel.value = level
  },
  onStatus: ({ connected: isConnected }) => {
    connected.value = isConnected
    log(isConnected ? 'connected' : 'disconnected')
  },
  onStop: () => {
    clearAll()
  },
  onError: (error) => {
    log(`error: ${error.message}`)
  },
  shouldMuteMic: () => isPlaying.value,
})

watch(
  [screenIntervalMs, cameraIntervalMs, diffThreshold, diffDownscaleWidth, diffDownscaleHeight, screenFrameRate, cameraFrameRate],
  () => {
    client.setVisionConfig({
      screenIntervalMs: Number(screenIntervalMs.value),
      cameraIntervalMs: Number(cameraIntervalMs.value),
      diffThreshold: Number(diffThreshold.value),
      diffDownscaleWidth: Number(diffDownscaleWidth.value),
      diffDownscaleHeight: Number(diffDownscaleHeight.value),
      screenFrameRate: Number(screenFrameRate.value),
      cameraFrameRate: Number(cameraFrameRate.value),
    })
  },
  { immediate: true },
)

async function handleConnect() {
  await client.connect({ url: wsUrl.value, sessionId: sessionId.value, userId: userId.value })
}

function handleDisconnect() {
  client.disconnect()
}

async function handleToggleMic() {
  micEnabled.value = !micEnabled.value
  await client.setMicEnabled(micEnabled.value)
}

async function handleToggleScreen() {
  screenEnabled.value = !screenEnabled.value
  await client.setScreenEnabled(screenEnabled.value)
}

async function handleToggleCamera() {
  cameraEnabled.value = !cameraEnabled.value
  await client.setCameraEnabled(cameraEnabled.value)
}

function handleSendText() {
  const text = inputText.value.trim()
  if (!text)
    return
  client.sendText(text)
  log(`user: ${text}`)
  inputText.value = ''
}

function handleKeydown(event: KeyboardEvent) {
  if (isEditableTarget(event.target))
    return
  if (matchesHotkey(event, screenHotkey.value)) {
    event.preventDefault()
    handleToggleScreen()
    return
  }
  if (matchesHotkey(event, cameraHotkey.value)) {
    event.preventDefault()
    handleToggleCamera()
  }
}

onMounted(() => {
  connectAudioContext(audioContext)
  if (!audioAnalyser.value) {
    audioAnalyser.value = audioContext.createAnalyser()
    connectAudioAnalyser(audioAnalyser.value)
  }
  window.addEventListener('keydown', handleKeydown)
})

onPlaybackStarted(() => {
  isPlaying.value = true
})

onPlaybackFinished(() => {
  isPlaying.value = false
})

onUnmounted(() => {
  client.disconnect()
  clearAll()
  window.removeEventListener('keydown', handleKeydown)
})
</script>

<template>
  <div :class="['p-4', 'space-y-4']">
    <div :class="['text-lg', 'font-600']">
      AIAvatarKit Bridge (WebSocket)
    </div>

    <div :class="['grid', 'gap-3', 'sm:grid-cols-2']">
      <label :class="['flex', 'flex-col', 'gap-1']">
        <span :class="['text-sm', 'text-neutral-500', 'dark:text-neutral-400']">WebSocket URL</span>
        <input v-model="wsUrl" :class="['rounded', 'border', 'border-neutral-300', 'px-2', 'py-1', 'text-sm', 'dark:border-neutral-700', 'dark:bg-neutral-900']">
      </label>
      <label :class="['flex', 'flex-col', 'gap-1']">
        <span :class="['text-sm', 'text-neutral-500', 'dark:text-neutral-400']">Session ID</span>
        <input v-model="sessionId" :class="['rounded', 'border', 'border-neutral-300', 'px-2', 'py-1', 'text-sm', 'dark:border-neutral-700', 'dark:bg-neutral-900']">
      </label>
      <label :class="['flex', 'flex-col', 'gap-1']">
        <span :class="['text-sm', 'text-neutral-500', 'dark:text-neutral-400']">User ID</span>
        <input v-model="userId" :class="['rounded', 'border', 'border-neutral-300', 'px-2', 'py-1', 'text-sm', 'dark:border-neutral-700', 'dark:bg-neutral-900']">
      </label>
      <div :class="['flex', 'items-end', 'gap-2']">
        <button :class="['rounded', 'border', 'border-neutral-300', 'px-3', 'py-1.5', 'text-sm', 'dark:border-neutral-700']" @click="handleConnect">
          Connect
        </button>
        <button :class="['rounded', 'border', 'border-neutral-300', 'px-3', 'py-1.5', 'text-sm', 'dark:border-neutral-700']" @click="handleDisconnect">
          Disconnect
        </button>
        <span :class="['text-xs', connected ? 'text-emerald-500' : 'text-neutral-400']">
          {{ connected ? 'online' : 'offline' }}
        </span>
      </div>
    </div>

    <div :class="['flex', 'flex-wrap', 'gap-2']">
      <button :class="['rounded', 'border', 'border-neutral-300', 'px-3', 'py-1.5', 'text-sm', 'dark:border-neutral-700']" @click="handleToggleMic">
        Mic: {{ micEnabled ? 'on' : 'off' }}
      </button>
      <button :class="['rounded', 'border', 'border-neutral-300', 'px-3', 'py-1.5', 'text-sm', 'dark:border-neutral-700']" @click="handleToggleScreen">
        Screen: {{ screenEnabled ? 'on' : 'off' }}
      </button>
      <button :class="['rounded', 'border', 'border-neutral-300', 'px-3', 'py-1.5', 'text-sm', 'dark:border-neutral-700']" @click="handleToggleCamera">
        Camera: {{ cameraEnabled ? 'on' : 'off' }}
      </button>
      <div :class="['text-xs', 'text-neutral-500', 'dark:text-neutral-400', 'self-center']">
        Mic level: {{ micLevel.toFixed(4) }}
      </div>
    </div>

    <div :class="['grid', 'gap-3', 'sm:grid-cols-2']">
      <label :class="['flex', 'flex-col', 'gap-1']">
        <span :class="['text-sm', 'text-neutral-500', 'dark:text-neutral-400']">Screen interval (ms)</span>
        <input v-model.number="screenIntervalMs" type="number" min="0" step="50" :class="['rounded', 'border', 'border-neutral-300', 'px-2', 'py-1', 'text-sm', 'dark:border-neutral-700', 'dark:bg-neutral-900']">
      </label>
      <label :class="['flex', 'flex-col', 'gap-1']">
        <span :class="['text-sm', 'text-neutral-500', 'dark:text-neutral-400']">Camera interval (ms)</span>
        <input v-model.number="cameraIntervalMs" type="number" min="0" step="50" :class="['rounded', 'border', 'border-neutral-300', 'px-2', 'py-1', 'text-sm', 'dark:border-neutral-700', 'dark:bg-neutral-900']">
      </label>
      <label :class="['flex', 'flex-col', 'gap-1']">
        <span :class="['text-sm', 'text-neutral-500', 'dark:text-neutral-400']">Diff threshold (0-1)</span>
        <input v-model.number="diffThreshold" type="number" min="0" max="1" step="0.01" :class="['rounded', 'border', 'border-neutral-300', 'px-2', 'py-1', 'text-sm', 'dark:border-neutral-700', 'dark:bg-neutral-900']">
      </label>
      <label :class="['flex', 'flex-col', 'gap-1']">
        <span :class="['text-sm', 'text-neutral-500', 'dark:text-neutral-400']">Downscale (W x H)</span>
        <div :class="['flex', 'gap-2']">
          <input v-model.number="diffDownscaleWidth" type="number" min="8" step="1" :class="['w-24', 'rounded', 'border', 'border-neutral-300', 'px-2', 'py-1', 'text-sm', 'dark:border-neutral-700', 'dark:bg-neutral-900']">
          <input v-model.number="diffDownscaleHeight" type="number" min="8" step="1" :class="['w-24', 'rounded', 'border', 'border-neutral-300', 'px-2', 'py-1', 'text-sm', 'dark:border-neutral-700', 'dark:bg-neutral-900']">
        </div>
      </label>
      <label :class="['flex', 'flex-col', 'gap-1']">
        <span :class="['text-sm', 'text-neutral-500', 'dark:text-neutral-400']">Screen FPS</span>
        <input v-model.number="screenFrameRate" type="number" min="1" step="1" :class="['rounded', 'border', 'border-neutral-300', 'px-2', 'py-1', 'text-sm', 'dark:border-neutral-700', 'dark:bg-neutral-900']">
      </label>
      <label :class="['flex', 'flex-col', 'gap-1']">
        <span :class="['text-sm', 'text-neutral-500', 'dark:text-neutral-400']">Camera FPS</span>
        <input v-model.number="cameraFrameRate" type="number" min="1" step="1" :class="['rounded', 'border', 'border-neutral-300', 'px-2', 'py-1', 'text-sm', 'dark:border-neutral-700', 'dark:bg-neutral-900']">
      </label>
    </div>

    <div :class="['grid', 'gap-3', 'sm:grid-cols-2']">
      <label :class="['flex', 'flex-col', 'gap-1']">
        <span :class="['text-sm', 'text-neutral-500', 'dark:text-neutral-400']">Hotkey: Toggle Screen</span>
        <input v-model="screenHotkey" placeholder="Ctrl+Shift+S" :class="['rounded', 'border', 'border-neutral-300', 'px-2', 'py-1', 'text-sm', 'dark:border-neutral-700', 'dark:bg-neutral-900']">
      </label>
      <label :class="['flex', 'flex-col', 'gap-1']">
        <span :class="['text-sm', 'text-neutral-500', 'dark:text-neutral-400']">Hotkey: Toggle Camera</span>
        <input v-model="cameraHotkey" placeholder="Ctrl+Shift+C" :class="['rounded', 'border', 'border-neutral-300', 'px-2', 'py-1', 'text-sm', 'dark:border-neutral-700', 'dark:bg-neutral-900']">
      </label>
    </div>

    <div :class="['flex', 'gap-2']">
      <input
        v-model="inputText"
        placeholder="Type text to send"
        :class="['flex-1', 'rounded', 'border', 'border-neutral-300', 'px-2', 'py-1', 'text-sm', 'dark:border-neutral-700', 'dark:bg-neutral-900']"
        @keydown.enter="handleSendText"
      >
      <button :class="['rounded', 'border', 'border-neutral-300', 'px-3', 'py-1.5', 'text-sm', 'dark:border-neutral-700']" @click="handleSendText">
        Send
      </button>
    </div>

    <div :class="['rounded', 'border', 'border-neutral-200', 'p-3', 'text-xs', 'h-56', 'overflow-auto', 'dark:border-neutral-800']">
      <div v-for="(line, index) in logs" :key="index" :class="['mb-1', 'whitespace-pre-wrap']">
        {{ line }}
      </div>
    </div>
  </div>
</template>

<route lang="yaml">
meta:
  layout: settings
  stageTransition:
    name: slide
</route>
