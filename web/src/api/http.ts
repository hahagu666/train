type StandardResponse<T> = {
  success: boolean
  message?: string
  code?: string
  data?: T
}

export class ApiError extends Error {
  status: number
  code?: string
  fields?: Record<string, string>
  raw?: unknown

  constructor(status: number, message: string, options?: { code?: string; fields?: Record<string, string>; raw?: unknown }) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = options?.code
    this.fields = options?.fields
    this.raw = options?.raw
  }
}

function isStandardResponse(value: unknown): value is StandardResponse<unknown> {
  return Boolean(value && typeof value === 'object' && 'success' in value)
}

function mapValidationFields(value: unknown): Record<string, string> | undefined {
  if (!Array.isArray(value)) return undefined
  const fields: Record<string, string> = {}
  for (const issue of value) {
    if (!issue || typeof issue !== 'object') continue
    const path = Array.isArray(issue.loc) ? issue.loc.filter((part: unknown) => part !== 'body').join('.') : ''
    if (path && typeof issue.msg === 'string') fields[path] = issue.msg
  }
  return Object.keys(fields).length ? fields : undefined
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(path, {
      ...init,
      headers: init?.body instanceof FormData
        ? init.headers
        : { 'Content-Type': 'application/json', ...init?.headers },
    })
  } catch (cause) {
    if ((cause as { name?: string })?.name === 'AbortError') throw cause
    throw new ApiError(0, '无法连接后端，请确认本地服务已启动', { raw: cause })
  }

  const body = await response.json().catch(() => null)
  if (!response.ok) {
    const wrapped = isStandardResponse(body) ? body : undefined
    throw new ApiError(response.status, wrapped?.message || `请求失败 (${response.status})`, {
      code: wrapped?.code,
      fields: mapValidationFields(wrapped?.data),
      raw: body,
    })
  }

  if (isStandardResponse(body)) {
    if (!body.success) {
      throw new ApiError(response.status, body.message || '请求失败', {
        code: body.code,
        raw: body,
      })
    }
    return body.data as T
  }

  return body as T
}
