import type { ChatServerEvent, ConnectionStatus } from './chat-events'

type SocketOptions = {
  sessionId: string
  onEvent: (event: ChatServerEvent) => void
  onStatus: (status: ConnectionStatus, reconnected: boolean) => void
}

export class ChatSocket {
  private socket?: WebSocket
  private heartbeat?: number
  private reconnectTimer?: number
  private reconnectAttempt = 0
  private manuallyClosed = false
  private connectedOnce = false

  private readonly options: SocketOptions

  constructor(options: SocketOptions) {
    this.options = options
  }

  connect() {
    this.manuallyClosed = false
    this.clearTimers()
    const reconnecting = this.connectedOnce || this.reconnectAttempt > 0
    this.options.onStatus(reconnecting ? 'reconnecting' : 'connecting', reconnecting)
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const url = `${protocol}//${window.location.host}/ws/chat/${encodeURIComponent(this.options.sessionId)}`
    this.socket = new WebSocket(url)

    this.socket.addEventListener('open', () => {
      const wasReconnect = this.connectedOnce
      this.connectedOnce = true
      this.reconnectAttempt = 0
      this.options.onStatus('connected', wasReconnect)
      this.heartbeat = window.setInterval(() => this.send({ type: 'ping' }), 25_000)
    })

    this.socket.addEventListener('message', (message) => {
      try {
        this.options.onEvent(JSON.parse(message.data) as ChatServerEvent)
      } catch {
        this.options.onEvent({ type: 'error', code: 'INVALID_EVENT', message: '收到无法识别的聊天事件' })
      }
    })

    this.socket.addEventListener('close', () => {
      this.clearTimers()
      if (this.manuallyClosed) return
      this.options.onStatus('reconnecting', true)
      const delay = Math.min(30_000, 1000 * 2 ** this.reconnectAttempt) + Math.floor(Math.random() * 400)
      this.reconnectAttempt += 1
      this.reconnectTimer = window.setTimeout(() => this.connect(), delay)
    })

    this.socket.addEventListener('error', () => this.socket?.close())
  }

  sendMessage(content: string) {
    return this.send({ type: 'message', content })
  }

  stop() {
    return this.send({ type: 'stop' })
  }

  close() {
    this.manuallyClosed = true
    this.clearTimers()
    this.socket?.close()
  }

  private send(payload: Record<string, string>) {
    if (this.socket?.readyState !== WebSocket.OPEN) return false
    this.socket.send(JSON.stringify(payload))
    return true
  }

  private clearTimers() {
    if (this.heartbeat) window.clearInterval(this.heartbeat)
    if (this.reconnectTimer) window.clearTimeout(this.reconnectTimer)
    this.heartbeat = undefined
    this.reconnectTimer = undefined
  }
}
