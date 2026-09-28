import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiError, request } from './http'

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('request', () => {
  it('unwraps a successful standard response', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({
      success: true,
      data: { value: 42 },
    }), { status: 200, headers: { 'Content-Type': 'application/json' } })))

    await expect(request<{ value: number }>('/api/test')).resolves.toEqual({ value: 42 })
  })

  it('returns a successful raw response', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify(['a', 'b']), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    })))

    await expect(request<string[]>('/api/test')).resolves.toEqual(['a', 'b'])
  })

  it('maps wrapped validation errors to fields', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({
      success: false,
      code: 'VALIDATION_ERROR',
      message: '请求参数无效',
      data: [{ loc: ['body', 'character_id'], msg: 'field required' }],
    }), { status: 422, headers: { 'Content-Type': 'application/json' } })))

    const error = await request('/api/test').catch((reason: unknown) => reason)
    expect(error).toBeInstanceOf(ApiError)
    expect(error).toMatchObject({
      status: 422,
      code: 'VALIDATION_ERROR',
      message: '请求参数无效',
      fields: { character_id: 'field required' },
    })
  })

  it('preserves AbortError for cancelled requests', async () => {
    const aborted = new DOMException('aborted', 'AbortError')
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(aborted))

    await expect(request('/api/test')).rejects.toBe(aborted)
  })

  it('normalizes network failures', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('offline')))

    const error = await request('/api/test').catch((reason: unknown) => reason)
    expect(error).toBeInstanceOf(ApiError)
    expect(error).toMatchObject({ status: 0, message: '无法连接后端，请确认本地服务已启动' })
  })
})
