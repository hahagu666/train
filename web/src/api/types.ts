export type HealthStatus = {
  status: 'ok' | 'loading'
  service: string
  model_loaded: boolean
}

export type ModelStatus = {
  main_model_loaded: boolean
  main_model_device: string
  main_model_quantization: string
  small_model_loaded: boolean
  small_model_available: boolean
  embedder_loaded: boolean
  loading: boolean
  load_progress: number
  main_model_state: string
  main_model_busy: boolean
  main_model_error: string
  main_model_device_map: string
  main_model_dtype: string
  main_model_footprint_bytes: number
  cuda_memory: Record<string, number>
  attention_backend: string
  use_cache: boolean
  last_generation: Record<string, number | string>
}

export type CharacterSummary = {
  id: string
  name: string
  avatar: string
  background: string
  avatar_revision: number
  background_revision: number
  is_preset: boolean
  relationship_type: string
  current_stage: string
  description_preview: string
}

export type CharacterDetail = CharacterSummary & {
  age?: number | null
  adult_verified: boolean
  sexual_interaction_allowed: boolean
  eligibility_reason: string
  character_description: string
  backstory: string
  initial_outfit: string
  initial_closeness: number
  initial_trust: number
  likes: string[]
  dislikes: string[]
  fears: string[]
  appearance: Record<string, unknown>
  personality: Record<string, unknown>
  speech_style: Record<string, unknown>
}

export type CharacterInput = {
  name: string
  age: number
  adult_confirmed: boolean
  relationship_type: string
  character_description: string
  backstory?: string
  initial_outfit?: string
  initial_closeness?: number
  initial_trust?: number
  likes?: string[]
  dislikes?: string[]
  fears?: string[]
}

export type CharacterUpdateInput = {
  name?: string
  age?: number
  adult_verified?: boolean
  character_description?: string
  personality_description?: string
  backstory?: string
  appearance?: Record<string, unknown>
  body_params?: Record<string, unknown>
  initial_outfit?: string
  likes?: string[]
  dislikes?: string[]
  fears?: string[]
}

export type ScenarioSummary = {
  id: string
  title: string
  stage: string
  stage_name: string
  location: string
  context_preview: string
}

export type ChatCommand = {
  id: string
  label: string
  description: string
  content: string
  group?: string
}

export type SessionMeta = {
  session_id: string
  character_id: string
  character_name: string
  session_name: string
  created_at: string
  last_active_at: string
  current_stage: string
  relationship_stage: string
  scenario_stage?: string | null
  total_interactions: number
  relationship_closeness: number
  relationship_trust: number
  summary: string
  is_active: boolean
  current_location: string
  current_time: string
  scenario_id?: string | null
}

export type ChatMessage = {
  message_id?: string
  client_id?: string
  pending?: boolean
  timestamp?: string
  role: 'user' | 'assistant' | 'system' | 'narrator' | string
  content: string
  turn?: number
  turn_id?: string
  timeline_id?: string
  snapshot_id?: string | null
}

export type SessionState = {
  arousal: number
  phase: string
  emotion: string
  trust: number
  closeness: number
  heart_rate: number
  clothing: string
  time_str: string
  location: string
  privacy: string
  privacy_level?: number
  danger_level?: number
  detection_risk?: number
  has_people?: boolean
  others_present?: boolean
  stage: string
  relationship_stage?: string
  scenario_stage?: string | null
  acceptance: number
  stamina: number
  fatigue: number
  is_orgasm: boolean
  is_interrupted: boolean
}

export type SessionUserProfile = {
  name: string
  gender: string
  age: number
  personality_description: string
  speaking_style_hint: string
  appearance_hint: string
  traits: string[]
  relation: string
  preferred_address: string
  additional_context: string
}

export type SessionUserProfileInput = Partial<SessionUserProfile>

export type SessionDetail = {
  meta: SessionMeta
  state: SessionState
  messages: ChatMessage[]
  scenario_id?: string | null
  snapshot_id?: string | null
  timeline_id?: string
  user_profile?: SessionUserProfile
}

export type CreateSessionInput = {
  character_id: string
  name?: string
  scenario_id?: string | null
  custom_opening?: string
  initial_trust?: number
  initial_closeness?: number
  initial_outfit?: string
  custom_outfit_description?: string
  location?: string
  time_str?: string
  privacy?: '私密' | '半公开' | '危险'
  user_profile?: SessionUserProfileInput
}

export type CreateSessionResult = {
  session_id: string
  meta: SessionMeta
  opening: string
  opening_message?: ChatMessage | null
  scenario_id?: string | null
  state: SessionState
  snapshot_id: string | null
  timeline_id: string
  user_profile: SessionUserProfile
}

export type ChatResponse = {
  response: string
  state: SessionState
  messages: ChatMessage[]
  job_id: string
  request_id: string
}

export type ChatStreamEvent = {
  type: string
  schema_version?: number
  sequence?: number
  job_id?: string
  request_id?: string
  session_id?: string
  content?: string
  full_response?: string
  response?: string
  data?: SessionState
  snapshot_id?: string | null
  turn?: number
  code?: string
  message?: string
  [key: string]: unknown
}

export type RetryResponse = ChatResponse & {
  snapshot_id: string | null
  timeline_id: string
}

export type GenerationStatus = {
  schema_version: number
  job_id: string
  request_id: string
  session_id: string
  kind: string
  status: 'queued' | 'running' | 'cancelling' | 'cancelled' | 'completed' | 'failed'
  queue_position: number | null
  queue_wait_ms: number | null
}

export type SessionStateDetail = {
  state: SessionState
  character: Record<string, unknown>
  world: Record<string, unknown>
  snapshot_id: string | null
  timeline_id: string
}

export type HistoryDetail = {
  messages: ChatMessage[]
  snapshots: Array<Record<string, unknown>>
  current_snapshot_id: string | null
  timeline_id: string
}

export type AppSettings = {
  version: number
  theme: 'system' | 'light' | 'dark'
  language: 'zh-CN'
  auto_save: boolean
  stream_responses: boolean
}

export type AppMenu = {
  characters: CharacterSummary[]
  sessions: SessionMeta[]
  active_session_id: string | null
  settings: AppSettings
}
