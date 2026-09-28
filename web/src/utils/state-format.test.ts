import { describe, expect, it } from 'vitest'
import { formatClothing } from './state-format'

describe('formatClothing', () => {
  it('localizes canonical clothing presets', () => {
    expect(formatClothing('pajamas')).toBe('睡衣')
    expect(formatClothing('DRESSED')).toBe('穿着整齐')
  })

  it('preserves natural-language clothing descriptions', () => {
    expect(formatClothing('穿着小熊图案睡衣上衣、小熊图案睡裤、白色棉质内裤')).toBe(
      '穿着小熊图案睡衣上衣、小熊图案睡裤、白色棉质内裤',
    )
  })

  it('uses the missing-state label only for blank values', () => {
    expect(formatClothing()).toBe('未记录')
    expect(formatClothing('   ')).toBe('未记录')
  })
})
