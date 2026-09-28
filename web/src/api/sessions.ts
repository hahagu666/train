import { request } from './http'
import type {
  ChatCommand,
  ChatResponse,
  ChatStreamEvent,
  CreateSessionInput,
  CreateSessionResult,
  HistoryDetail,
  RetryResponse,
  ScenarioSummary,
  GenerationStatus,
  SessionDetail,
  SessionMeta,
  SessionStateDetail,
} from './types'

export const sessionsApi = {
  list: (characterId?: string, signal?: AbortSignal) => {
    const params = characterId ? `?char_id=${encodeURIComponent(characterId)}` : ''
    return request<SessionMeta[]>(`/api/sessions${params}`, { signal })
  },
  detail: (sessionId: string, signal?: AbortSignal) =>
    request<SessionDetail>(`/api/sessions/${encodeURIComponent(sessionId)}`, { signal }),
  history: (sessionId: string, signal?: AbortSignal) =>
    request<HistoryDetail>(`/api/sessions/${encodeURIComponent(sessionId)}/history`, { signal }),
  state: (sessionId: string, signal?: AbortSignal) =>
    request<SessionStateDetail>(`/api/sessions/${encodeURIComponent(sessionId)}/state`, { signal }),
  generation: (sessionId: string, signal?: AbortSignal) =>
    request<GenerationStatus | null>(`/api/generation/sessions/${encodeURIComponent(sessionId)}`, { signal }),
  scenarios: (signal?: AbortSignal) => request<ScenarioSummary[]>('/api/scenarios', { signal }),
  commands: (sessionId: string, signal?: AbortSignal) =>
    request<ChatCommand[]>(`/api/chat/commands?session_id=${encodeURIComponent(sessionId)}`, { signal }),
  chat: (sessionId: string, message: string, requestId: string, signal?: AbortSignal) =>
    request<ChatResponse>('/api/chat', {
      method: 'POST',
      body: JSON.stringify({ session_id: sessionId, message, request_id: requestId }),
      signal,
    }),
  chatStream: async function* (sessionId: string, message: string, requestId: string, signal?: AbortSignal) {
    const response = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/x-ndjson' },
      body: JSON.stringify({ session_id: sessionId, message, request_id: requestId }),
      signal,
    })
    if (!response.ok || !response.body) {
      const body = await response.text().catch(() => '')
      throw new Error(body || `请求失败 (${response.status})`)
    }
    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    try {
      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''
        for (const line of lines) {
          if (line.trim()) yield JSON.parse(line) as ChatStreamEvent
        }
      }
      buffer += decoder.decode()
      if (buffer.trim()) yield JSON.parse(buffer) as ChatStreamEvent
    } finally {
      reader.releaseLock()
    }
  },
  retry: (sessionId: string, turn: number, requestId: string, suggestion = '', signal?: AbortSignal) =>
    request<RetryResponse>(
      `/api/sessions/${encodeURIComponent(sessionId)}/retry?turn=${encodeURIComponent(turn)}`,
      {
        method: 'POST',
        body: JSON.stringify({ suggestion, request_id: requestId }),
        signal,
      },
    ),
  cancelChat: (sessionId: string, requestId: string) =>
    request<{ status: 'cancelling' | 'finished' }>(
      `/api/chat/${encodeURIComponent(sessionId)}/cancel/${encodeURIComponent(requestId)}`,
      { method: 'POST' },
    ),
  create: (input: CreateSessionInput) =>
    request<CreateSessionResult>('/api/sessions', {
      method: 'POST',
      body: JSON.stringify(input),
    }),
  delete: (sessionId: string) =>
    request<void>(`/api/sessions/${encodeURIComponent(sessionId)}`, { method: 'DELETE' }),
  autosave: (sessionId: string) =>
    request<Record<string, unknown>>(`/api/sessions/${encodeURIComponent(sessionId)}/autosave`, { method: 'PUT' }),
}
