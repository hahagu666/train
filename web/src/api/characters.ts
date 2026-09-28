import { request } from './http'
import type { CharacterDetail, CharacterInput, CharacterSummary, CharacterUpdateInput } from './types'

export const charactersApi = {
  list: (signal?: AbortSignal) => request<CharacterSummary[]>('/api/characters', { signal }),
  detail: (characterId: string, signal?: AbortSignal) =>
    request<CharacterDetail>(`/api/characters/${encodeURIComponent(characterId)}`, { signal }),
  create: (input: CharacterInput) =>
    request<CharacterDetail>('/api/characters', {
      method: 'POST',
      body: JSON.stringify(input),
    }),
  update: (characterId: string, input: CharacterUpdateInput) =>
    request<CharacterDetail>(`/api/characters/${encodeURIComponent(characterId)}`, {
      method: 'PATCH',
      body: JSON.stringify(input),
    }),
  delete: (characterId: string) =>
    request<void>(`/api/characters/${encodeURIComponent(characterId)}`, { method: 'DELETE' }),
}

export function characterMediaUrl(characterId: string, kind: 'avatar' | 'background', revision: number) {
  return `/api/media/${encodeURIComponent(characterId)}/${kind}?v=${revision}`
}
