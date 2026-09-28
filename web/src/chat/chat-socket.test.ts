// @vitest-environment jsdom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ChatSocket } from './chat-socket'
import type { ChatServerEvent, ConnectionStatus } from './chat-events'

class FakeWebSocket {
  static instances: FakeWebSocket[] = []
  static readonly CONNECTING = 0
  static readonly OPEN = 1
  static readonly CLOSING = 2
  static readonly CLOSED = 3

  readonly url: string
  readyState = FakeWebSocket.CONNECTING
  sent: string[] = []
  private listeners = new Map<string, Array<(event: { data?: string }) => void>>()

  constructor(url: string) {
    this.url = url
    FakeWebSocket.instances.push(this)
  }

  addEventListener(type: string, listener: (event: { data?: string }) => void) {
    const listeners = this.listeners.get(type) || []
    listeners.push(listener)
    this.listeners.set(type, listeners)
  }

  send(data: string) {
    this.sent.push(data)
  }

  close() {
    this.readyState = FakeWebSocket.CLOSED
    this.emit('close')
  }

  open() {
    this.readyState = FakeWebSocket.OPEN
    this.emit('open')
  }

  emit(type: string, event: { data?: string } = {}) {
    for (const listener of this.listeners.get(type) || []) listener(event)
  }
}

describe('ChatSocket', () => {
  let statuses: Array<[ConnectionStatus, boolean]>
  let events: ChatServerEvent[]

  beforeEach(() => {
    vi.useFakeTimers()
    FakeWebSocket.instances = []
    statuses = []
    events = []
    vi.stubGlobal('WebSocket', FakeWebSocket)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.useRealTimers()
  })

  function createSocket() {
    return new ChatSocket({
      sessionId: 'session/1',
      onEvent: (event) => events.push(event),
      onStatus: (status, reconnected) => statuses.push([status, reconnected]),
    })
  }

  it('connects and sends each chat command once', () => {
    const socket = createSocket()
    socket.connect()

    const webSocket = FakeWebSocket.instances[0]
    expect(webSocket.url).toBe('ws://localhost:3000/ws/chat/session%2F1')
    expect(statuses).toEqual([['connecting', false]])

    webSocket.open()
    expect(statuses.at(-1)).toEqual(['connected', false])
    expect(socket.sendMessage('你好')).toBe(true)
    expect(socket.stop()).toBe(true)
    expect(webSocket.sent).toEqual([
      JSON.stringify({ type: 'message', content: '你好' }),
      JSON.stringify({ type: 'stop' }),
    ])
  })

  it('does not reconnect or emit a stale status after manual close', () => {
    const socket = createSocket()
    socket.connect()
    FakeWebSocket.instances[0].open()

    socket.close()
    vi.runAllTimers()

    expect(FakeWebSocket.instances).toHaveLength(1)
    expect(statuses).toEqual([['connecting', false], ['connected', false]])
  })

  it('reconnects after an unexpected close and marks the restored connection', () => {
    vi.spyOn(Math, 'random').mockReturnValue(0)
    const socket = createSocket()
    socket.connect()
    FakeWebSocket.instances[0].open()

    FakeWebSocket.instances[0].emit('close')
    expect(statuses.at(-1)).toEqual(['reconnecting', true])

    vi.advanceTimersByTime(1_000)
    expect(FakeWebSocket.instances).toHaveLength(2)
    FakeWebSocket.instances[1].open()

    expect(statuses.at(-1)).toEqual(['connected', true])
  })

  it('reports malformed server events', () => {
    const socket = createSocket()
    socket.connect()
    FakeWebSocket.instances[0].emit('message', { data: '{invalid' })

    expect(events).toEqual([{ type: 'error', code: 'INVALID_EVENT', message: '收到无法识别的聊天事件' }])
  })
})
