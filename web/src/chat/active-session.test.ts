import { afterEach, describe, expect, it, vi } from 'vitest'
import { loadActiveSessionId, saveActiveSessionId } from './active-session'

function storage(initial: Record<string, string> = {}) {
  const values = new Map(Object.entries(initial))
  return {
    getItem: vi.fn((key: string) => values.get(key) ?? null),
    setItem: vi.fn((key: string, value: string) => values.set(key, value)),
    removeItem: vi.fn((key: string) => values.delete(key)),
  }
}

afterEach(() => vi.unstubAllGlobals())

describe('active session persistence', () => {
  it('restores and updates the selected session', () => {
    const localStorage = storage({ 'heart-chat.active-session-id': 'session-old' })
    vi.stubGlobal('window', { localStorage })

    expect(loadActiveSessionId()).toBe('session-old')
    saveActiveSessionId('session-new')
    expect(localStorage.setItem).toHaveBeenCalledWith('heart-chat.active-session-id', 'session-new')
  })

  it('clears selection and tolerates unavailable storage', () => {
    const localStorage = storage()
    vi.stubGlobal('window', { localStorage })

    saveActiveSessionId()
    expect(localStorage.removeItem).toHaveBeenCalledWith('heart-chat.active-session-id')

    vi.stubGlobal('window', { localStorage: { getItem: () => { throw new Error('blocked') } } })
    expect(loadActiveSessionId()).toBeUndefined()
    expect(() => saveActiveSessionId('session')).not.toThrow()
  })
})
