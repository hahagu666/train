import { describe, expect, it } from 'vitest'
import { buildCreateSessionPayload, emptySessionProfile, type CreateSessionFormValues } from './session-payload'

const automaticValues = (): CreateSessionFormValues => ({
  characterId: 'character-1',
  name: '',
  scenarioId: '',
  customOpening: '',
  initialTrust: '',
  initialCloseness: '',
  outfit: { mode: 'automatic', value: '' },
  location: { mode: 'automatic', value: '' },
  timeStr: '',
  privacy: '',
  userProfile: emptySessionProfile(),
})

describe('buildCreateSessionPayload', () => {
  it('omits every automatic or blank optional field', () => {
    expect(buildCreateSessionPayload(automaticValues())).toEqual({ character_id: 'character-1' })
  })

  it('includes only explicitly supplied overrides and trims natural text', () => {
    expect(buildCreateSessionPayload({
      ...automaticValues(),
      name: '  雨夜  ',
      scenarioId: ' scene-1 ',
      customOpening: '  她在等我。 ',
      initialTrust: '0',
      initialCloseness: '0.35',
      outfit: { mode: 'custom', value: '  宽松的白色针织衫  ' },
      location: { mode: 'custom', value: '  安静的客厅  ' },
      timeStr: ' 深夜 ',
      privacy: '半公开',
    })).toEqual({
      character_id: 'character-1',
      name: '雨夜',
      scenario_id: 'scene-1',
      custom_opening: '她在等我。',
      initial_trust: 0,
      initial_closeness: 0.35,
      custom_outfit_description: '宽松的白色针织衫',
      location: '安静的客厅',
      time_str: '深夜',
      privacy: '半公开',
    })
  })

  it('does not leak stale custom text while its mode is automatic', () => {
    expect(buildCreateSessionPayload({
      ...automaticValues(),
      outfit: { mode: 'automatic', value: '旧衣着' },
      location: { mode: 'automatic', value: '旧地点' },
    })).toEqual({ character_id: 'character-1' })
  })

  it('sends predefined and custom outfit modes through distinct fields', () => {
    expect(buildCreateSessionPayload({
      ...automaticValues(),
      outfit: { mode: 'preset', value: ' pajamas ' },
    })).toEqual({ character_id: 'character-1', initial_outfit: 'pajamas' })
    expect(buildCreateSessionPayload({
      ...automaticValues(),
      outfit: { mode: 'custom', value: '  白色针织衫 ' },
    })).toEqual({ character_id: 'character-1', custom_outfit_description: '白色针织衫' })
  })

  it('normalizes a partial per-session profile override', () => {
    expect(buildCreateSessionPayload({
      ...automaticValues(),
      userProfile: {
        ...emptySessionProfile(),
        name: '  小林 ',
        age: ' 24 ',
        personalityDescription: ' 安静认真 ',
        traits: '细心、耐心，细心\n可靠',
        preferredAddress: ' 前辈 ',
      },
    })).toEqual({
      character_id: 'character-1',
      user_profile: {
        name: '小林',
        age: 24,
        personality_description: '安静认真',
        traits: ['细心', '耐心', '可靠'],
        preferred_address: '前辈',
      },
    })
  })

  it('omits invalid optional numeric text rather than emitting NaN', () => {
    expect(buildCreateSessionPayload({
      ...automaticValues(),
      initialTrust: 'not-a-number',
      userProfile: { ...emptySessionProfile(), age: '121' },
    })).toEqual({ character_id: 'character-1' })
  })
})
