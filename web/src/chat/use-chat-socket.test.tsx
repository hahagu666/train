// @vitest-environment jsdom

import { act, renderHook, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { ChatMessage, ChatStreamEvent } from '../api/types'
import type { SetStateAction } from 'react'
import { sessionsApi } from '../api/sessions'
import { useChatSocket } from './use-chat-socket'

vi.mock('../api/sessions', () => ({
  sessionsApi: {
    chatStream: vi.fn(),
    retry: vi.fn(),
    cancelChat: vi.fn(),
    generation: vi.fn(),
  },
}))

function createMessageHarness() {
  let messages: ChatMessage[] = []
  const setMessages = vi.fn((update: SetStateAction<ChatMessage[]>) => {
    messages = typeof update === 'function' ? update(messages) : update
  })
  return { setMessages, getMessages: () => messages }
}

function stream(events: ChatStreamEvent[]) {
  return (async function* () {
    for (const event of events) yield event
  })()
}

function deferredStream() {
  let release: ((event: ChatStreamEvent) => void) | undefined
  let finished = false
  const events: ChatStreamEvent[] = []
  const waiters: Array<(event: ChatStreamEvent) => void> = []
  const iterable = (async function* () {
    while (!finished) {
      const event = events.shift() || await new Promise<ChatStreamEvent>((resolve) => waiters.push(resolve))
      if (event.type === '__close__') break
      yield event
    }
  })()
  release = (event) => {
    const waiter = waiters.shift()
    if (waiter) waiter(event)
    else events.push(event)
  }
  return { iterable, push: release, close: () => { finished = true; release?.({ type: '__close__' }) } }
}

describe('useChatSocket NDJSON streaming', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(sessionsApi.generation).mockResolvedValue(null)
  })

  it('applies identity-matched incremental events and commits the response', async () => {
    vi.mocked(sessionsApi.chatStream).mockImplementation((sessionId, _message, requestId) => stream([
      { type: 'generating', session_id: sessionId, request_id: requestId, job_id: 'job', sequence: 1 },
      { type: 'token', content: '你', session_id: sessionId, request_id: requestId, job_id: 'job', sequence: 2 },
      { type: 'token', content: '好', session_id: sessionId, request_id: requestId, job_id: 'job', sequence: 3 },
      { type: 'completed', response: '你好', session_id: sessionId, request_id: requestId, job_id: 'job', sequence: 4 },
    ]))
    const messages = createMessageHarness()
    const onResync = vi.fn().mockResolvedValue(undefined)
    const { result } = renderHook(() => useChatSocket({ sessionId: 'session', onMessages: messages.setMessages, onResync }))

    act(() => { expect(result.current.send('问候')).toBe(true) })
    await waitFor(() => expect(result.current.chat.generation).toBe('idle'))

    expect(messages.getMessages()).toEqual([
      expect.objectContaining({ role: 'user', content: '问候', pending: true }),
      expect.objectContaining({ role: 'assistant', content: '你好', pending: true }),
    ])
    expect(result.current.chat.streamBuffer).toBe('你好')
    expect(onResync).toHaveBeenCalledWith('session')
  })

  it('ignores stale identities and duplicate sequences', async () => {
    vi.mocked(sessionsApi.chatStream).mockImplementation((sessionId, _message, requestId) => stream([
      { type: 'token', content: '错', session_id: 'other', request_id: requestId, sequence: 1 },
      { type: 'token', content: '对', session_id: sessionId, request_id: requestId, job_id: 'job', sequence: 2 },
      { type: 'token', content: '重', session_id: sessionId, request_id: requestId, job_id: 'job', sequence: 2 },
      { type: 'completed', response: '对', session_id: sessionId, request_id: requestId, job_id: 'job', sequence: 3 },
    ]))
    const messages = createMessageHarness()
    const { result } = renderHook(() => useChatSocket({
      sessionId: 'session', onMessages: messages.setMessages, onResync: vi.fn().mockResolvedValue(undefined),
    }))

    act(() => { result.current.send('消息') })
    await waitFor(() => expect(result.current.chat.generation).toBe('idle'))
    expect(result.current.chat.streamBuffer).toBe('对')
  })

  it('keeps background session generation while switching sessions', async () => {
    const oldStream = deferredStream()
    const newStream = deferredStream()
    vi.mocked(sessionsApi.chatStream)
      .mockReturnValueOnce(oldStream.iterable)
      .mockReturnValueOnce(newStream.iterable)
    const messages = createMessageHarness()
    const onResync = vi.fn().mockResolvedValue(undefined)
    const { result, rerender } = renderHook(
      ({ sessionId }) => useChatSocket({ sessionId, onMessages: messages.setMessages, onResync }),
      { initialProps: { sessionId: 'old' } },
    )

    act(() => { expect(result.current.send('旧消息')).toBe(true) })
    rerender({ sessionId: 'new' })
    act(() => { expect(result.current.send('新消息')).toBe(true) })
    expect(result.current.chat.generation).toBe('thinking')

    oldStream.push({ type: 'completed', response: '旧回复', session_id: 'old', sequence: 1 })
    oldStream.close()
    await act(async () => { await Promise.resolve() })
    expect(result.current.chat.generation).toBe('thinking')

    rerender({ sessionId: 'old' })
    await waitFor(() => expect(result.current.chat.generation).toBe('idle'))
    expect(result.current.chat.streamBuffer).toBe('旧回复')
    newStream.close()
  })

  it('uses explicit backend cancellation before aborting transport', async () => {
    const pending = deferredStream()
    let signal: AbortSignal | undefined
    vi.mocked(sessionsApi.chatStream).mockImplementation((_session, _message, _request, value) => {
      signal = value
      return pending.iterable
    })
    vi.mocked(sessionsApi.cancelChat).mockResolvedValue({ status: 'cancelling' })
    const messages = createMessageHarness()
    const { result } = renderHook(() => useChatSocket({
      sessionId: 'session', onMessages: messages.setMessages, onResync: vi.fn().mockResolvedValue(undefined),
    }))

    act(() => {
      result.current.send('等待停止')
      expect(result.current.stop()).toBe(true)
    })
    expect(result.current.chat.generation).toBe('stopping')
    expect(signal?.aborted).toBe(false)
    expect(sessionsApi.cancelChat).toHaveBeenCalledWith('session', expect.any(String))

    pending.push({ type: 'cancelled', message: '已停止', session_id: 'session', sequence: 1 })
    pending.close()
    await waitFor(() => expect(result.current.chat.generation).toBe('idle'))
  })

  it('retains partial tokens when generation ends with an error', async () => {
    vi.mocked(sessionsApi.chatStream).mockImplementation((sessionId, _message, requestId) => stream([
      { type: 'generating', session_id: sessionId, request_id: requestId, job_id: 'job', sequence: 1 },
      { type: 'token', content: '不会消失', session_id: sessionId, request_id: requestId, job_id: 'job', sequence: 2 },
      { type: 'error', message: '生成超时', session_id: sessionId, request_id: requestId, job_id: 'job', sequence: 3 },
    ]))
    const messages = createMessageHarness()
    const { result } = renderHook(() => useChatSocket({
      sessionId: 'session', onMessages: messages.setMessages, onResync: vi.fn().mockResolvedValue(undefined),
    }))

    act(() => { result.current.send('继续') })
    await waitFor(() => expect(result.current.chat.generation).toBe('idle'))
    expect(result.current.chat.streamBuffer).toBe('不会消失')
    expect(result.current.chat.lastError).toBe('生成超时')
  })

  it('polls an active job after premature EOF and resyncs its final response', async () => {
    vi.mocked(sessionsApi.chatStream).mockImplementation((sessionId, _message, requestId) => stream([
      { type: 'generating', session_id: sessionId, request_id: requestId, job_id: 'job-eof', sequence: 1 },
      { type: 'token', content: '部分回复', session_id: sessionId, request_id: requestId, job_id: 'job-eof', sequence: 2 },
    ]))
    vi.mocked(sessionsApi.generation)
      .mockResolvedValueOnce(null)
      .mockResolvedValueOnce({
        schema_version: 1,
        job_id: 'job-eof',
        request_id: 'request-eof',
        session_id: 'session',
        kind: 'chat',
        status: 'running',
        queue_position: null,
        queue_wait_ms: 0,
      })
      .mockResolvedValueOnce(null)
    const messages = createMessageHarness()
    const onResync = vi.fn().mockImplementation(async () => {
      messages.setMessages([
        { role: 'user', content: '复杂动作', pending: false },
        { role: 'assistant', content: '后台持久化的完整回复', pending: false },
      ])
    })
    const { result } = renderHook(() => useChatSocket({
      sessionId: 'session', onMessages: messages.setMessages, onResync,
    }))

    act(() => { expect(result.current.send('复杂动作')).toBe(true) })
    await waitFor(() => expect(result.current.chat.connection).toBe('reconnecting'))
    expect(result.current.chat.streamBuffer).toBe('部分回复')
    expect(onResync).not.toHaveBeenCalled()

    await waitFor(() => expect(onResync).toHaveBeenCalledWith('session'), { timeout: 1500 })
    expect(result.current.chat.generation).toBe('idle')
    expect(messages.getMessages()).toEqual([
      expect.objectContaining({ role: 'user', content: '复杂动作', pending: false }),
      expect.objectContaining({ role: 'assistant', content: '后台持久化的完整回复', pending: false }),
    ])
  })

  it('keeps the optimistic user message when transport fails but backend completes', async () => {
    vi.mocked(sessionsApi.chatStream).mockImplementation((sessionId, _message, requestId) => (async function* () {
      yield { type: 'token', content: '临时片段', session_id: sessionId, request_id: requestId, job_id: 'job-error', sequence: 1 }
      throw new Error('网络连接中断')
    })())
    vi.mocked(sessionsApi.generation)
      .mockResolvedValueOnce(null)
      .mockResolvedValueOnce({
        schema_version: 1,
        job_id: 'job-error',
        request_id: 'request-error',
        session_id: 'session',
        kind: 'chat',
        status: 'running',
        queue_position: null,
        queue_wait_ms: 0,
      })
      .mockResolvedValueOnce(null)
    const messages = createMessageHarness()
    const onResync = vi.fn().mockResolvedValue(undefined)
    const { result } = renderHook(() => useChatSocket({
      sessionId: 'session', onMessages: messages.setMessages, onResync,
    }))

    act(() => { expect(result.current.send('不要丢失')).toBe(true) })
    await waitFor(() => expect(result.current.chat.connection).toBe('reconnecting'))
    expect(result.current.chat.lastError).toBe('网络连接中断')

    await waitFor(() => expect(onResync).toHaveBeenCalledWith('session'), { timeout: 1500 })
    expect(messages.getMessages()).toEqual([
      expect.objectContaining({ role: 'user', content: '不要丢失', pending: true }),
    ])
    expect(result.current.chat.lastError).toBeUndefined()
  })

  it('resumes an active generation after refresh and resyncs when it finishes', async () => {
    vi.mocked(sessionsApi.generation)
      .mockResolvedValueOnce({
        schema_version: 1,
        job_id: 'job-resume',
        request_id: 'request-resume',
        session_id: 'session',
        kind: 'chat',
        status: 'running',
        queue_position: null,
        queue_wait_ms: 12,
      })
      .mockResolvedValueOnce(null)
    const onResync = vi.fn().mockResolvedValue(undefined)
    const messages = createMessageHarness()
    const { result } = renderHook(() => useChatSocket({
      sessionId: 'session', onMessages: messages.setMessages, onResync,
    }))

    await waitFor(() => expect(onResync).toHaveBeenCalledWith('session'), { timeout: 1500 })
    expect(result.current.chat.generation).toBe('idle')
    expect(sessionsApi.generation).toHaveBeenLastCalledWith('session', expect.any(AbortSignal))
  })

  it('retries and resyncs the originating session', async () => {
    vi.mocked(sessionsApi.retry).mockResolvedValue({} as never)
    const onResync = vi.fn().mockResolvedValue(undefined)
    const messages = createMessageHarness()
    const { result } = renderHook(() => useChatSocket({ sessionId: 'session', onMessages: messages.setMessages, onResync }))

    act(() => { expect(result.current.retry(2, '更温柔')).toBe(true) })
    await waitFor(() => expect(result.current.chat.generation).toBe('idle'))
    expect(sessionsApi.retry).toHaveBeenCalledWith('session', 2, expect.any(String), '更温柔', expect.any(AbortSignal))
    expect(onResync).toHaveBeenCalledWith('session')
  })
})
