import { describe, expect, it, vi } from 'vitest'
import { applyQuickCommand } from './quick-command'

describe('applyQuickCommand', () => {
  it('fills the draft, closes the panel, and focuses without sending', () => {
    const setDraft = vi.fn()
    const close = vi.fn()
    const focus = vi.fn()
    const send = vi.fn()

    applyQuickCommand('先聊聊今天吧', { setDraft, close, focus })

    expect(setDraft).toHaveBeenCalledWith('先聊聊今天吧')
    expect(close).toHaveBeenCalledOnce()
    expect(focus).toHaveBeenCalledOnce()
    expect(send).not.toHaveBeenCalled()
  })
})
