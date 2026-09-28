import { useCallback, useEffect, useReducer, useRef, type Dispatch, type SetStateAction } from 'react'
import type { ChatMessage, ChatStreamEvent } from '../api/types'
import { sessionsApi } from '../api/sessions'
import {
  chatReducer,
  commitAssistantResponse,
  initialChatState,
  removePendingMessage,
  type ChatAction,
  type ChatServerEvent,
  type ChatState,
} from './chat-events'

type UseChatSocketOptions = {
  sessionId?: string
  onMessages: Dispatch<SetStateAction<ChatMessage[]>>
  onResync: (sessionId?: string) => Promise<void>
}

type ActiveRequest = {
  controller: AbortController
  operation: 'send' | 'retry' | 'resume'
  requestId: string
  clientId?: string
  sessionId: string
  stopRequested: boolean
  jobId?: string
  lastSequence: number
}

type SessionStates = Record<string, ChatState>
type SessionAction = { sessionId: string; action: ChatAction }

function reduceSessionState(states: SessionStates, update: SessionAction): SessionStates {
  return {
    ...states,
    [update.sessionId]: chatReducer(states[update.sessionId] || initialChatState, update.action),
  }
}

function createRequestId() {
  return globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`
}

function errorMessage(error: unknown, fallback: string) {
  return error instanceof Error ? error.message : fallback
}

function isAbortError(error: unknown) {
  return (error as { name?: string } | undefined)?.name === 'AbortError'
}

const ACTIVE_GENERATION_STATUSES = new Set(['queued', 'running', 'cancelling'])
const RECOVERY_POLL_INTERVAL_MS = 1_000
const RECOVERY_TIMEOUT_MS = 6 * 60_000

function delay(milliseconds: number) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds))
}

function acceptsEvent(active: ActiveRequest, event: ChatStreamEvent) {
  if (event.session_id && event.session_id !== active.sessionId) return false
  if (event.request_id && event.request_id !== active.requestId) return false
  if (active.jobId && event.job_id && event.job_id !== active.jobId) return false
  if (!active.jobId && event.job_id) active.jobId = event.job_id
  if (typeof event.sequence === 'number') {
    if (event.sequence <= active.lastSequence) return false
    active.lastSequence = event.sequence
  }
  return true
}

function asServerEvent(event: ChatStreamEvent): ChatServerEvent | undefined {
  switch (event.type) {
    case 'opening':
      return { type: 'opening', content: String(event.content || '') }
    case 'action_start':
      return { type: 'action_start', action: typeof event.action === 'string' ? event.action : undefined }
    case 'blocked':
      return { type: 'blocked', code: event.code, message: event.message }
    case 'orgasm':
      return { type: 'orgasm', intensity: typeof event.intensity === 'number' ? event.intensity : undefined }
    case 'event':
      return {
        type: 'event',
        event_type: typeof event.event_type === 'string' ? event.event_type : undefined,
        desc: typeof event.desc === 'string' ? event.desc : undefined,
      }
    case 'generating':
      return { type: 'generating' }
    case 'token':
      return { type: 'token', content: String(event.content || '') }
    case 'memory_formed':
      return { type: 'memory_formed', summary: typeof event.summary === 'string' ? event.summary : undefined }
    case 'state':
      return event.data ? { type: 'state', data: event.data, snapshot_id: event.snapshot_id || undefined } : undefined
    case 'done':
      return { type: 'done', full_response: String(event.full_response || ''), snapshot_id: event.snapshot_id || undefined, turn: event.turn }
    case 'completed':
      return { type: 'completed', response: String(event.response || event.full_response || '') }
    case 'cancelled':
      return { type: 'cancelled', message: event.message }
    case 'error':
      return { type: 'error', code: event.code, message: event.message }
    case 'pong':
      return { type: 'pong' }
    default:
      return undefined
  }
}

export function useChatSocket({ sessionId, onMessages, onResync }: UseChatSocketOptions) {
  const [states, dispatchSession] = useReducer(reduceSessionState, {})
  const activeRef = useRef(new Map<string, ActiveRequest>())
  const mountedRef = useRef(true)
  const selectedSessionRef = useRef(sessionId)
  selectedSessionRef.current = sessionId

  const dispatchFor = useCallback((targetSessionId: string, action: ChatAction) => {
    dispatchSession({ sessionId: targetSessionId, action })
  }, [])

  const recoverGeneration = useCallback(async (active: ActiveRequest) => {
    const deadline = Date.now() + RECOVERY_TIMEOUT_MS
    while (mountedRef.current && activeRef.current.get(active.sessionId) === active) {
      const generation = await sessionsApi.generation(active.sessionId, active.controller.signal)
      if (!generation || !ACTIVE_GENERATION_STATUSES.has(generation.status)) {
        dispatchFor(active.sessionId, { type: 'reset-transient' })
        await onResync(active.sessionId)
        dispatchFor(active.sessionId, { type: 'reset-transient' })
        return
      }
      if (generation.status === 'cancelling') {
        active.stopRequested = true
        dispatchFor(active.sessionId, { type: 'stopping' })
      } else {
        dispatchFor(active.sessionId, { type: 'connection', value: 'reconnecting' })
      }
      if (Date.now() >= deadline) throw new Error('等待后台生成完成超时，请刷新会话重试')
      await delay(RECOVERY_POLL_INTERVAL_MS)
    }
  }, [dispatchFor, onResync])

  useEffect(() => {
    if (!sessionId || activeRef.current.has(sessionId)) return
    const controller = new AbortController()
    let disposed = false
    void sessionsApi.generation(sessionId, controller.signal).then((generation) => {
      if (disposed || !generation || !ACTIVE_GENERATION_STATUSES.has(generation.status)) return
      if (activeRef.current.has(sessionId)) return
      const active: ActiveRequest = {
        controller,
        operation: 'resume',
        requestId: generation.request_id,
        sessionId,
        stopRequested: generation.status === 'cancelling',
        jobId: generation.job_id,
        lastSequence: 0,
      }
      activeRef.current.set(sessionId, active)
      dispatchFor(sessionId, { type: generation.status === 'cancelling' ? 'stopping' : 'sending' })
      void recoverGeneration(active).catch((error) => {
        if (mountedRef.current && !isAbortError(error)) {
          dispatchFor(sessionId, { type: 'server', event: { type: 'error', message: errorMessage(error, '会话同步失败') } })
        }
      }).finally(() => {
        if (activeRef.current.get(sessionId) === active) activeRef.current.delete(sessionId)
      })
    }).catch((error) => {
      if (!disposed && !isAbortError(error)) {
        dispatchFor(sessionId, { type: 'server', event: { type: 'error', message: errorMessage(error, '生成状态同步失败') } })
      }
    })
    return () => {
      disposed = true
      if (activeRef.current.get(sessionId)?.operation !== 'resume') controller.abort()
    }
  }, [dispatchFor, recoverGeneration, sessionId])

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      for (const active of activeRef.current.values()) active.controller.abort()
      activeRef.current.clear()
    }
  }, [])

  const send = useCallback((content: string) => {
    if (!sessionId || activeRef.current.has(sessionId)) return false
    const active: ActiveRequest = {
      controller: new AbortController(),
      operation: 'send',
      requestId: createRequestId(),
      clientId: createRequestId(),
      sessionId,
      stopRequested: false,
      lastSequence: 0,
    }
    activeRef.current.set(sessionId, active)
    onMessages((messages) => [...messages, {
      client_id: active.clientId,
      role: 'user', content, pending: true, timestamp: new Date().toISOString(),
    }])
    dispatchFor(sessionId, { type: 'sending' })

    void (async () => {
      let requestError: unknown
      let terminal = false
      let recovered = false
      try {
        for await (const rawEvent of sessionsApi.chatStream(
          sessionId, content, active.requestId, active.controller.signal,
        )) {
          if (!acceptsEvent(active, rawEvent)) continue
          const event = asServerEvent(rawEvent)
          if (!event) continue
          dispatchFor(sessionId, { type: 'server', event })
          if (event.type === 'done') {
            terminal = true
            if (selectedSessionRef.current === sessionId) {
              onMessages((messages) => commitAssistantResponse(messages, event.full_response))
            }
          } else if (event.type === 'completed') {
            terminal = true
            if (event.response && selectedSessionRef.current === sessionId) {
              onMessages((messages) => commitAssistantResponse(messages, event.response))
            }
          } else if (event.type === 'cancelled' || event.type === 'error') {
            terminal = true
          }
        }
      } catch (error) {
        requestError = error
        if (!isAbortError(error)) {
          dispatchFor(sessionId, { type: 'server', event: { type: 'error', message: errorMessage(error, '聊天处理失败') } })
        }
      }

      if (terminal) {
        if (!isAbortError(requestError)) {
          try {
            await onResync(sessionId)
          } catch (error) {
            dispatchFor(sessionId, { type: 'server', event: { type: 'error', message: errorMessage(error, '会话同步失败') } })
          }
        }
        dispatchFor(sessionId, { type: 'commit-complete' })
      } else if (!isAbortError(requestError)) {
        try {
          await recoverGeneration(active)
          recovered = true
        } catch (error) {
          if (!isAbortError(error)) {
            dispatchFor(sessionId, { type: 'server', event: { type: 'error', message: errorMessage(error, '会话同步失败') } })
          }
        }
      }
      if (requestError && !recovered && selectedSessionRef.current === sessionId && !active.stopRequested && !isAbortError(requestError)) {
        onMessages((messages) => removePendingMessage(messages, active.clientId!))
      }
    })().finally(() => {
      if (activeRef.current.get(sessionId) !== active) return
      activeRef.current.delete(sessionId)
    })
    return true
  }, [dispatchFor, onMessages, onResync, recoverGeneration, sessionId])

  const retry = useCallback((turn: number, suggestion = '') => {
    if (!sessionId || activeRef.current.has(sessionId) || !Number.isInteger(turn) || turn <= 0) return false
    const active: ActiveRequest = {
      controller: new AbortController(),
      operation: 'retry',
      requestId: createRequestId(),
      sessionId,
      stopRequested: false,
      lastSequence: 0,
    }
    activeRef.current.set(sessionId, active)
    dispatchFor(sessionId, { type: 'sending' })
    void (async () => {
      let requestError: unknown
      try {
        await sessionsApi.retry(sessionId, turn, active.requestId, suggestion, active.controller.signal)
      } catch (error) {
        requestError = error
      }
      if (!isAbortError(requestError) || active.stopRequested) {
        try {
          await onResync(sessionId)
        } catch (error) {
          dispatchFor(sessionId, { type: 'server', event: { type: 'error', message: errorMessage(error, '会话同步失败') } })
          return
        }
      }
      if (requestError && !active.stopRequested && !isAbortError(requestError)) {
        dispatchFor(sessionId, { type: 'server', event: { type: 'error', message: errorMessage(requestError, '重说失败') } })
      }
    })().finally(() => {
      if (activeRef.current.get(sessionId) !== active) return
      activeRef.current.delete(sessionId)
      dispatchFor(sessionId, { type: 'reset-transient' })
    })
    return true
  }, [dispatchFor, onResync, sessionId])

  const stop = useCallback(() => {
    if (!sessionId) return false
    const active = activeRef.current.get(sessionId)
    if (!active || active.stopRequested) return false
    active.stopRequested = true
    dispatchFor(sessionId, { type: 'stopping' })
    void sessionsApi.cancelChat(active.sessionId, active.requestId).catch(() => {
      active.controller.abort()
    })
    return true
  }, [dispatchFor, sessionId])

  return { chat: sessionId ? states[sessionId] || initialChatState : initialChatState, send, retry, stop }
}

export async function resyncSession(sessionId: string) {
  const [detail, history, state, generation] = await Promise.all([
    sessionsApi.detail(sessionId), sessionsApi.history(sessionId), sessionsApi.state(sessionId),
    sessionsApi.generation(sessionId),
  ])
  return { detail, history, state, generation }
}
