<script setup lang="ts">
import { createAiAvatarKitClient } from '@proj-airi/stage-ui/libs'
import { usePipelineCharacterSpeechPlaybackQueueStore } from '@proj-airi/stage-ui/composables/queues'
import { EMOTION_EmotionMotionName_value, Emotion } from '@proj-airi/stage-ui/constants/emotions'
import { useAudioContext } from '@proj-airi/stage-ui/stores/audio'
import { useLive2d } from '@proj-airi/stage-ui/stores/live2d'
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
const { currentMotion } = storeToRefs(live2dStore)

const audioAnalyser = ref<AnalyserNode>()

function log(line: string) {
  logs.value = [line, ...logs.value].slice(0, 80)
}

function mapFaceToMotion(face: string) {
  const key = face.toLowerCase()
  const emotion = ({
    neutral: Emotion.Idle,
    joy: Emotion.Happy,
    happy: Emotion.Happy,
    angry: Emotion.Angry,
    sorrow: Emotion.Sad,
    sad: Emotion.Sad,
    fun: Emotion.Happy,
    surprised: Emotion.Surprise,
  } as Record<string, Emotion>)[key] ?? Emotion.Idle

  const motion = EMOTION_EmotionMotionName_value[emotion]
  if (motion)
    currentMotion.value = { group: motion }
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
    mapFaceToMotion(face)
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
