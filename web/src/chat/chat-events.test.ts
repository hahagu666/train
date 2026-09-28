import { describe, expect, it } from 'vitest'
import type { ChatMessage, SessionState } from '../api/types'
import { chatReducer, commitAssistantResponse, initialChatState, mergeOpening, reconcileMessages, removePendingMessage } from './chat-events'

const stateSnapshot = {
  arousal: 0.2,
  phase: 'excitement',
  emotion: '平静',
  trust: 0.4,
  closeness: 0.3,
  heart_rate: 76,
  clothing: '常服',
  time_str: '上午',
  location: '客厅',
  privacy: '较私密',
  stage: '1',
  acceptance: 0.25,
  stamina: 0.9,
  fatigue: 0.1,
  is_orgasm: false,
  is_interrupted: false,
} satisfies SessionState

describe('chatReducer', () => {
  it('moves through send and token generation states', () => {
    const sending = chatReducer(initialChatState, { type: 'sending' })
    expect(sending).toMatchObject({ generation: 'thinking', streamBuffer: '' })

    const firstToken = chatReducer(sending, { type: 'server', event: { type: 'token', content: '你' } })
    const secondToken = chatReducer(firstToken, { type: 'server', event: { type: 'token', content: '好' } })
    expect(secondToken).toMatchObject({ generation: 'generating', streamBuffer: '你好' })
  })

  it('stores authoritative state snapshots', () => {
    const next = chatReducer(initialChatState, { type: 'server', event: { type: 'state', data: stateSnapshot } })
    expect(next.state).toEqual(stateSnapshot)
  })

  it('retains an interrupted partial response after cancellation and reset', () => {
    const cancelled = chatReducer(
      { ...initialChatState, generation: 'stopping', streamBuffer: 'partial' },
      { type: 'server', event: { type: 'cancelled', message: '已停止' } },
    )
    expect(cancelled).toMatchObject({ generation: 'idle', streamBuffer: 'partial', lastError: '已停止' })

    const reset = chatReducer(cancelled, { type: 'reset-transient' })
    expect(reset).toMatchObject({ generation: 'idle', streamBuffer: '', notices: [], lastError: undefined })
  })

  it('retains an interrupted partial response after an error', () => {
    const failed = chatReducer(
      { ...initialChatState, generation: 'generating', streamBuffer: '已经生成的部分' },
      { type: 'server', event: { type: 'error', message: '生成超时' } },
    )
    expect(failed).toMatchObject({ generation: 'idle', streamBuffer: '已经生成的部分', lastError: '生成超时' })
  })

  it('clears all session-specific state when switching sessions', () => {
    const state = chatReducer({
      ...initialChatState,
      connection: 'connected',
      generation: 'generating',
      streamBuffer: '旧会话回复',
      notices: [{ id: 'notice', tone: 'warning', text: '旧会话提示' }],
      lastError: '旧会话错误',
      state: stateSnapshot,
    }, { type: 'reset-session' })

    expect(state).toEqual(initialChatState)
  })

  it('keeps only the four latest process notices', () => {
    let state = initialChatState
    for (let index = 0; index < 6; index += 1) {
      state = chatReducer(state, { type: 'server', event: { type: 'event', desc: `事件 ${index}` } })
    }
    expect(state.notices.map((notice) => notice.text)).toEqual(['事件 2', '事件 3', '事件 4', '事件 5'])
  })
})

describe('mergeOpening', () => {
  it('adds an opening message once', () => {
    const messages: ChatMessage[] = []
    const withOpening = mergeOpening(messages, '你好')
    expect(withOpening).toHaveLength(1)
    expect(withOpening[0]).toMatchObject({ role: 'assistant', content: '你好' })
    expect(mergeOpening(withOpening, ' 你好 ')).toBe(withOpening)
  })

  it('ignores an empty opening', () => {
    const messages: ChatMessage[] = []
    expect(mergeOpening(messages, '   ')).toBe(messages)
  })
})

describe('message reconciliation', () => {
  it('keeps an optimistic user message until the server commits it', () => {
    const optimistic: ChatMessage = { role: 'user', content: '想你了', client_id: 'client-1', pending: true }
    expect(reconcileMessages([optimistic], [])).toEqual([optimistic])

    const committed: ChatMessage = { message_id: 'server-1', role: 'user', content: '想你了' }
    expect(reconcileMessages([optimistic], [committed])).toEqual([{ ...committed, pending: false }])
  })

  it('does not drop a repeated optimistic message because older history has the same text', () => {
    const prior: ChatMessage = { message_id: 'old', role: 'user', content: '继续' }
    const optimistic: ChatMessage = { role: 'user', content: '继续', client_id: 'client-2', pending: true }
    expect(reconcileMessages([prior, optimistic], [prior])).toEqual([{ ...prior, pending: false }, optimistic])

    const committedAgain: ChatMessage = { message_id: 'new', role: 'user', content: '继续' }
    expect(reconcileMessages([prior, optimistic], [prior, committedAgain])).toEqual([
      { ...prior, pending: false },
      { ...committedAgain, pending: false },
    ])
  })

  it('removes only the cancelled request optimistic message', () => {
    const cancelled: ChatMessage = { role: 'user', content: '停止这一条', client_id: 'cancelled', pending: true }
    const other: ChatMessage = { role: 'user', content: '保留这一条', client_id: 'other', pending: true }
    const committed: ChatMessage = { message_id: 'server', role: 'assistant', content: '已提交' }

    expect(removePendingMessage([committed, cancelled, other], 'cancelled')).toEqual([committed, other])
  })

  it('commits a done response before REST history catches up', () => {
    const messages = commitAssistantResponse([], '（抬起头）我也想你。')
    expect(messages).toEqual([
      expect.objectContaining({ role: 'assistant', content: '（抬起头）我也想你。', pending: true }),
    ])
    expect(commitAssistantResponse(messages, '（抬起头）我也想你。')).toBe(messages)
  })
})
