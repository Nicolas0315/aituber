export interface AiAvatarKitAudioFormat {
  sample_rate: number
  channels: number
  sample_width: number
}

export interface AiAvatarKitAvatarControl {
  face_name?: string | null
  face_duration?: number | null
  animation_name?: string | null
  animation_duration?: number | null
}

export interface AiAvatarKitVisionConfig {
  screenIntervalMs: number
  cameraIntervalMs: number
  diffThreshold: number
  diffDownscaleWidth: number
  diffDownscaleHeight: number
  screenFrameRate: number
  cameraFrameRate: number
}

export interface AiAvatarKitResponse {
  type: string
  session_id?: string
  user_id?: string
  context_id?: string
  text?: string | null
  voice_text?: string | null
  audio_data?: string | null
  avatar_control_request?: AiAvatarKitAvatarControl | null
  metadata?: Record<string, any> | null
}

export interface AiAvatarKitClientOptions {
  audioContext: AudioContext
  enqueueAudio: (payload: { audioBuffer: AudioBuffer, text: string }) => void
  onText?: (text: string, raw: AiAvatarKitResponse) => void
  onFace?: (face: string, duration?: number) => void
  onAnimation?: (name: string, duration?: number) => void
  onMicLevel?: (level: number) => void
  onStatus?: (status: { connected: boolean }) => void
  onStop?: () => void
  onError?: (error: Error) => void
  shouldMuteMic?: () => boolean
  vision?: Partial<AiAvatarKitVisionConfig>
}

interface FrameGrabber {
  start: () => Promise<void>
  stop: () => void
  capture: () => Promise<string | null>
  updateConfig: (config: Partial<VisionCaptureConfig>) => void
}

interface VisionCaptureConfig {
  minIntervalMs: number
  diffThreshold: number
  diffDownscaleWidth: number
  diffDownscaleHeight: number
}

function arrayBufferToBase64(buffer: ArrayBuffer) {
  const bytes = new Uint8Array(buffer)
  let binary = ''
  for (let i = 0; i < bytes.byteLength; i++)
    binary += String.fromCharCode(bytes[i])
  return btoa(binary)
}

function base64ToArrayBuffer(base64: string) {
  const binary = atob(base64)
  const len = binary.length
  const bytes = new Uint8Array(len)
  for (let i = 0; i < len; i++)
    bytes[i] = binary.charCodeAt(i)
  return bytes.buffer
}

function float32ToInt16Buffer(floatBuffer: Float32Array) {
  const buffer = new ArrayBuffer(floatBuffer.length * 2)
  const view = new DataView(buffer)
  for (let i = 0; i < floatBuffer.length; i++) {
    let sample = floatBuffer[i]
    sample = Math.max(-1, Math.min(1, sample))
    const intSample = sample < 0 ? sample * 32768 : sample * 32767
    view.setInt16(i * 2, intSample, true)
  }
  return buffer
}

function createFrameGrabber(streamFactory: () => Promise<MediaStream>, config: VisionCaptureConfig): FrameGrabber {
  let stream: MediaStream | null = null
  let video: HTMLVideoElement | null = null
  let lastSample: Uint8ClampedArray | null = null
  let lastCaptureAt = 0

  const canvas = document.createElement('canvas')
  const sampleCanvas = document.createElement('canvas')
  const ctx = canvas.getContext('2d')
  const sampleCtx = sampleCanvas.getContext('2d')

  function updateConfig(next: Partial<VisionCaptureConfig>) {
    Object.assign(config, next)
  }

  async function start() {
    if (stream)
      return

    stream = await streamFactory()
    video = document.createElement('video')
    video.srcObject = stream
    video.muted = true
    await video.play()
  }

  function stop() {
    if (stream) {
      stream.getTracks().forEach(track => track.stop())
      stream = null
    }
    if (video) {
      video.pause()
      video.srcObject = null
      video = null
    }
  }

  function computeSample(width: number, height: number) {
    if (!sampleCtx)
      return null
    const sampleWidth = Math.max(8, Math.floor(config.diffDownscaleWidth))
    const sampleHeight = Math.max(8, Math.floor(config.diffDownscaleHeight))
    sampleCanvas.width = sampleWidth
    sampleCanvas.height = sampleHeight
    sampleCtx.drawImage(canvas, 0, 0, width, height, 0, 0, sampleWidth, sampleHeight)
    const data = sampleCtx.getImageData(0, 0, sampleWidth, sampleHeight).data
    const luma = new Uint8ClampedArray(sampleWidth * sampleHeight)
    for (let i = 0, j = 0; i < data.length; i += 4, j++) {
      const r = data[i]
      const g = data[i + 1]
      const b = data[i + 2]
      luma[j] = Math.round((0.2126 * r) + (0.7152 * g) + (0.0722 * b))
    }
    return luma
  }

  function diffRatio(sample: Uint8ClampedArray, prev: Uint8ClampedArray) {
    if (sample.length !== prev.length)
      return 1
    let sum = 0
    for (let i = 0; i < sample.length; i++)
      sum += Math.abs(sample[i] - prev[i])
    return sum / (sample.length * 255)
  }

  async function capture() {
    if (!video || !ctx)
      return null

    const now = Date.now()
    if (config.minIntervalMs > 0 && (now - lastCaptureAt) < config.minIntervalMs)
      return null

    const width = video.videoWidth || 1280
    const height = video.videoHeight || 720
    canvas.width = width
    canvas.height = height
    ctx.drawImage(video, 0, 0, width, height)

    const sample = computeSample(width, height)
    if (sample && config.diffThreshold > 0 && lastSample) {
      const delta = diffRatio(sample, lastSample)
      if (delta < config.diffThreshold) {
        lastCaptureAt = now
        return null
      }
    }

    if (sample)
      lastSample = sample
    lastCaptureAt = now
    return canvas.toDataURL('image/jpeg', 0.9)
  }

  return { start, stop, capture, updateConfig }
}

export function createAiAvatarKitClient(options: AiAvatarKitClientOptions) {
  let ws: WebSocket | null = null
  let sessionId = ''
  let userId = ''
  let contextId: string | null = null
  let micStream: MediaStream | null = null
  let micSource: MediaStreamAudioSourceNode | null = null
  let scriptNode: ScriptProcessorNode | null = null
  let screenGrabber: FrameGrabber | null = null
  let cameraGrabber: FrameGrabber | null = null
  const visionConfig: AiAvatarKitVisionConfig = {
    screenIntervalMs: 1000,
    cameraIntervalMs: 1000,
    diffThreshold: 0.08,
    diffDownscaleWidth: 64,
    diffDownscaleHeight: 36,
    screenFrameRate: 5,
    cameraFrameRate: 5,
    ...options.vision,
  }

  const state = {
    connected: false,
    micEnabled: false,
    screenEnabled: false,
    cameraEnabled: false,
  }

  function setConnected(next: boolean) {
    state.connected = next
    options.onStatus?.({ connected: next })
  }

  function sendMessage(payload: Record<string, any>) {
    if (ws && ws.readyState === WebSocket.OPEN)
      ws.send(JSON.stringify(payload))
  }

  function setVisionConfig(next: Partial<AiAvatarKitVisionConfig>) {
    Object.assign(visionConfig, next)
    screenGrabber?.updateConfig({
      minIntervalMs: visionConfig.screenIntervalMs,
      diffThreshold: visionConfig.diffThreshold,
      diffDownscaleWidth: visionConfig.diffDownscaleWidth,
      diffDownscaleHeight: visionConfig.diffDownscaleHeight,
    })
    cameraGrabber?.updateConfig({
      minIntervalMs: visionConfig.cameraIntervalMs,
      diffThreshold: visionConfig.diffThreshold,
      diffDownscaleWidth: visionConfig.diffDownscaleWidth,
      diffDownscaleHeight: visionConfig.diffDownscaleHeight,
    })
  }

  async function connect(params: { url: string, sessionId: string, userId: string }) {
    await options.audioContext.resume()
    sessionId = params.sessionId
    userId = params.userId

    ws = new WebSocket(params.url)
    ws.onopen = () => {
      setConnected(true)
      contextId = null
      sendMessage({ type: 'start', session_id: sessionId, user_id: userId, context_id: null })
    }
    ws.onerror = (event) => {
      options.onError?.(new Error(`WebSocket error: ${String(event)}`))
    }
    ws.onclose = () => {
      setConnected(false)
    }
    ws.onmessage = async (event) => {
      let msg: AiAvatarKitResponse | null = null
      try {
        msg = JSON.parse(event.data)
      }
      catch (err) {
        options.onError?.(err as Error)
        return
      }

      if (!msg)
        return

      if (msg.type === 'stop') {
        options.onStop?.()
        return
      }

      if (msg.type === 'start' && msg.context_id) {
        contextId = msg.context_id
        return
      }

      if (msg.type === 'vision') {
        const source = msg.metadata?.source
        await handleVisionRequest(source)
        return
      }

      if (msg.type === 'chunk') {
        if (msg.text)
          options.onText?.(msg.text, msg)

        const face = msg.avatar_control_request?.face_name
        if (face)
          options.onFace?.(face, msg.avatar_control_request?.face_duration ?? undefined)

        const animation = msg.avatar_control_request?.animation_name
        if (animation)
          options.onAnimation?.(animation, msg.avatar_control_request?.animation_duration ?? undefined)

        if (msg.audio_data) {
          const audioBuffer = await decodeAudio(msg.audio_data, msg.metadata?.pcm_format)
          if (audioBuffer)
            options.enqueueAudio({ audioBuffer, text: msg.voice_text || msg.text || '' })
        }
      }
    }
  }

  async function decodeAudio(audioDataBase64: string, pcmFormat?: AiAvatarKitAudioFormat) {
    const buffer = base64ToArrayBuffer(audioDataBase64)
    if (pcmFormat) {
      if (pcmFormat.sample_width !== 2)
        throw new Error('Only 16-bit PCM is supported')
      const sampleCount = buffer.byteLength / 2 / pcmFormat.channels
      const audioBuffer = options.audioContext.createBuffer(
        pcmFormat.channels,
        sampleCount,
        pcmFormat.sample_rate,
      )
      const pcm = new Int16Array(buffer)
      for (let ch = 0; ch < pcmFormat.channels; ch++) {
        const channelData = audioBuffer.getChannelData(ch)
        for (let i = 0; i < sampleCount; i++) {
          channelData[i] = pcm[i * pcmFormat.channels + ch] / 32768
        }
      }
      return audioBuffer
    }

    return await options.audioContext.decodeAudioData(buffer.slice(0))
  }

  async function startMic() {
    if (micStream)
      return
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
    })
    micSource = options.audioContext.createMediaStreamSource(micStream)
    scriptNode = options.audioContext.createScriptProcessor(256, 1, 1)
    scriptNode.onaudioprocess = (event) => {
      const inputData = event.inputBuffer.getChannelData(0)
      const pcmBuffer = float32ToInt16Buffer(inputData)
      if (!state.connected || !state.micEnabled)
        return
      if (options.shouldMuteMic?.())
        return

      let sum = 0
      for (let i = 0; i < inputData.length; i++)
        sum += inputData[i] * inputData[i]
      const rms = Math.sqrt(sum / inputData.length)
      options.onMicLevel?.(rms)

      sendMessage({
        type: 'data',
        session_id: sessionId,
        audio_data: arrayBufferToBase64(pcmBuffer),
      })
    }
    micSource.connect(scriptNode)
    scriptNode.connect(options.audioContext.destination)
  }

  function stopMic() {
    if (scriptNode)
      scriptNode.disconnect()
    if (micSource)
      micSource.disconnect()
    if (micStream)
      micStream.getTracks().forEach(track => track.stop())
    scriptNode = null
    micSource = null
    micStream = null
  }

  async function setMicEnabled(enabled: boolean) {
    state.micEnabled = enabled
    if (enabled)
      await startMic()
    else
      stopMic()
  }

  async function setScreenEnabled(enabled: boolean) {
    state.screenEnabled = enabled
    if (enabled) {
      if (!screenGrabber) {
        screenGrabber = createFrameGrabber(
          () => navigator.mediaDevices.getDisplayMedia({
            video: { frameRate: visionConfig.screenFrameRate },
            audio: false,
          }),
          {
            minIntervalMs: visionConfig.screenIntervalMs,
            diffThreshold: visionConfig.diffThreshold,
            diffDownscaleWidth: visionConfig.diffDownscaleWidth,
            diffDownscaleHeight: visionConfig.diffDownscaleHeight,
          },
        )
      }
      await screenGrabber.start()
    }
    else {
      screenGrabber?.stop()
    }
  }

  async function setCameraEnabled(enabled: boolean) {
    state.cameraEnabled = enabled
    if (enabled) {
      if (!cameraGrabber) {
        cameraGrabber = createFrameGrabber(
          () => navigator.mediaDevices.getUserMedia({
            video: { frameRate: visionConfig.cameraFrameRate },
            audio: false,
          }),
          {
            minIntervalMs: visionConfig.cameraIntervalMs,
            diffThreshold: visionConfig.diffThreshold,
            diffDownscaleWidth: visionConfig.diffDownscaleWidth,
            diffDownscaleHeight: visionConfig.diffDownscaleHeight,
          },
        )
      }
      await cameraGrabber.start()
    }
    else {
      cameraGrabber?.stop()
    }
  }

  async function handleVisionRequest(source?: string) {
    let imageUrl: string | null = null
    if (source === 'camera' && state.cameraEnabled && cameraGrabber)
      imageUrl = await cameraGrabber.capture()
    else if (source === 'screenshot' && state.screenEnabled && screenGrabber)
      imageUrl = await screenGrabber.capture()

    if (!imageUrl)
      return

    sendMessage({
      type: 'invoke',
      session_id: sessionId,
      user_id: userId,
      context_id: contextId,
      files: [{ type: 'image', url: imageUrl }],
    })
  }

  function sendText(text: string) {
    sendMessage({
      type: 'invoke',
      session_id: sessionId,
      user_id: userId,
      context_id: contextId,
      text,
    })
  }

  function disconnect() {
    setConnected(false)
    contextId = null
    stopMic()
    screenGrabber?.stop()
    cameraGrabber?.stop()
    if (ws && ws.readyState === WebSocket.OPEN)
      sendMessage({ type: 'stop', session_id: sessionId })
    ws?.close()
    ws = null
  }

  return {
    connect,
    disconnect,
    sendText,
    setMicEnabled,
    setScreenEnabled,
    setCameraEnabled,
    setVisionConfig,
    state,
  }
}
