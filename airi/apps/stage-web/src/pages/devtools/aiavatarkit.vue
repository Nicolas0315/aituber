<script setup lang="ts">
import { createAiAvatarKitClient } from '@proj-airi/stage-ui/libs'
import { usePipelineCharacterSpeechPlaybackQueueStore } from '@proj-airi/stage-ui/composables/queues'
import { EMOTION_EmotionMotionName_value, Emotion } from '@proj-airi/stage-ui/constants/emotions'
import { useAudioContext } from '@proj-airi/stage-ui/stores/audio'
import { defaultModelParameters, useLive2d } from '@proj-airi/stage-ui/stores/live2d'
import { storeToRefs } from 'pinia'
import { onMounted, onUnmounted, ref } from 'vue'

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

onMounted(() => {
  connectAudioContext(audioContext)
  if (!audioAnalyser.value) {
    audioAnalyser.value = audioContext.createAnalyser()
    connectAudioAnalyser(audioAnalyser.value)
  }
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
