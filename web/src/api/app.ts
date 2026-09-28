import { request } from './http'
import type { AppMenu, AppSettings, HealthStatus, ModelStatus } from './types'

export const appApi = {
  health: (signal?: AbortSignal) => request<HealthStatus>('/api/health', { signal }),
  modelStatus: (signal?: AbortSignal) => request<ModelStatus>('/api/model/status', { signal }),
  menu: (signal?: AbortSignal) => request<AppMenu>('/api/app/menu', { signal }),
  settings: (signal?: AbortSignal) => request<AppSettings>('/api/app/settings', { signal }),
}
