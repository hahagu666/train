import { afterEach, describe, expect, it, vi } from 'vitest'
import { sessionsApi } from './sessions'

afterEach(() => vi.unstubAllGlobals())

describe('sessionsApi', () => {
  it('requests commands for the active session', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify([]), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    }))
    vi.stubGlobal('fetch', fetchMock)

    await sessionsApi.commands('session/一')

    expect(fetchMock).toHaveBeenCalledWith('/api/chat/commands?session_id=session%2F%E4%B8%80', expect.any(Object))
  })

  it('requests generation status for refresh resynchronization', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({
      success: true,
      data: { status: 'running', session_id: 'session/一' },
    }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    }))
    vi.stubGlobal('fetch', fetchMock)

    await sessionsApi.generation('session/一')

    expect(fetchMock).toHaveBeenCalledWith(
      '/api/generation/sessions/session%2F%E4%B8%80',
      expect.any(Object),
    )
  })

  it('parses NDJSON split across transport chunks', async () => {
    const encoder = new TextEncoder()
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode('{"type":"token","content":"你"}\n{"type":"tok'))
        controller.enqueue(encoder.encode('en","content":"好"}\n'))
        controller.close()
      },
    })
    const fetchMock = vi.fn().mockResolvedValue(new Response(body, {
      status: 200,
      headers: { 'Content-Type': 'application/x-ndjson' },
    }))
    vi.stubGlobal('fetch', fetchMock)

    const events = []
    for await (const event of sessionsApi.chatStream('session', '问候', 'request')) events.push(event)

    expect(events).toEqual([
      { type: 'token', content: '你' },
      { type: 'token', content: '好' },
    ])
    expect(fetchMock).toHaveBeenCalledWith('/api/chat/stream', expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({ session_id: 'session', message: '问候', request_id: 'request' }),
    }))
  })

  it('posts retry with its request ID, suggestion, turn, and abort signal', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({
      response: 'new reply',
      messages: [],
      state: {},
      snapshot_id: 'snapshot-2',
      timeline_id: 'timeline-2',
    }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    }))
    vi.stubGlobal('fetch', fetchMock)
    const controller = new AbortController()

    await sessionsApi.retry('session/一', 2, 'request-2', '换一种表达', controller.signal)

    expect(fetchMock).toHaveBeenCalledWith(
      '/api/sessions/session%2F%E4%B8%80/retry?turn=2',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ suggestion: '换一种表达', request_id: 'request-2' }),
        signal: controller.signal,
      }),
    )
  })
})
