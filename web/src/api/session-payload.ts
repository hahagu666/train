import type { CreateSessionInput } from './types'

export type AutomaticOrCustomText = {
  mode: 'automatic' | 'preset' | 'custom'
  value: string
}

export type SessionProfileFormValues = {
  name: string
  gender: string
  age: string
  personalityDescription: string
  speakingStyleHint: string
  appearanceHint: string
  traits: string
  relation: string
  preferredAddress: string
  additionalContext: string
}

export type CreateSessionFormValues = {
  characterId: string
  name: string
  scenarioId: string
  customOpening: string
  initialTrust: string
  initialCloseness: string
  outfit: AutomaticOrCustomText
  location: AutomaticOrCustomText
  timeStr: string
  privacy: '' | NonNullable<CreateSessionInput['privacy']>
  userProfile: SessionProfileFormValues
}

export const emptySessionProfile = (): SessionProfileFormValues => ({
  name: '',
  gender: '',
  age: '',
  personalityDescription: '',
  speakingStyleHint: '',
  appearanceHint: '',
  traits: '',
  relation: '',
  preferredAddress: '',
  additionalContext: '',
})

const trimmed = (value: string) => value.trim()

const normalizedTraits = (value: string) => {
  const result: string[] = []
  for (const part of value.split(/[、,，\n]/)) {
    const trait = part.trim()
    if (trait && trait.length <= 80 && !result.includes(trait)) result.push(trait)
    if (result.length >= 30) break
  }
  return result
}

export function buildCreateSessionPayload(values: CreateSessionFormValues): CreateSessionInput {
  const payload: CreateSessionInput = { character_id: values.characterId }
  const name = trimmed(values.name)
  const scenarioId = trimmed(values.scenarioId)
  const customOpening = trimmed(values.customOpening)
  const initialTrust = trimmed(values.initialTrust)
  const initialCloseness = trimmed(values.initialCloseness)
  const outfit = trimmed(values.outfit.value)
  const location = trimmed(values.location.value)
  const timeStr = trimmed(values.timeStr)

  if (name) payload.name = name
  if (scenarioId) payload.scenario_id = scenarioId
  if (customOpening) payload.custom_opening = customOpening
  if (initialTrust && Number.isFinite(Number(initialTrust))) payload.initial_trust = Number(initialTrust)
  if (initialCloseness && Number.isFinite(Number(initialCloseness))) payload.initial_closeness = Number(initialCloseness)
  if (values.outfit.mode === 'preset' && outfit) payload.initial_outfit = outfit
  if (values.outfit.mode === 'custom' && outfit) payload.custom_outfit_description = outfit
  if (values.location.mode === 'custom' && location) payload.location = location
  if (timeStr) payload.time_str = timeStr
  if (values.privacy) payload.privacy = values.privacy

  const profile = values.userProfile
  const profileText = {
    name: trimmed(profile.name),
    gender: trimmed(profile.gender),
    personality_description: trimmed(profile.personalityDescription),
    speaking_style_hint: trimmed(profile.speakingStyleHint),
    appearance_hint: trimmed(profile.appearanceHint),
    relation: trimmed(profile.relation),
    preferred_address: trimmed(profile.preferredAddress),
    additional_context: trimmed(profile.additionalContext),
  }
  const userProfile: NonNullable<CreateSessionInput['user_profile']> = {}
  for (const [key, value] of Object.entries(profileText)) {
    if (value) userProfile[key as keyof typeof profileText] = value
  }
  const age = trimmed(profile.age)
  if (age && Number.isInteger(Number(age)) && Number(age) >= 0 && Number(age) <= 120) userProfile.age = Number(age)
  const traits = normalizedTraits(profile.traits)
  if (traits.length) userProfile.traits = traits
  if (Object.keys(userProfile).length) payload.user_profile = userProfile

  return payload
}
