import { useCallback, useEffect, useMemo, useRef, useState, type Dispatch, type FormEvent, type SetStateAction } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Activity, ChevronDown, ChevronRight, CircleAlert, Edit3, Heart, HeartPulse, Menu,
  MessageSquarePlus, PanelRight, Plus, RefreshCw, Search, Send, Square, Trash2, X,
} from 'lucide-react'
import { ApiError } from './api/http'
import { appApi } from './api/app'
import { characterMediaUrl, charactersApi } from './api/characters'
import { sessionsApi } from './api/sessions'
import { buildCreateSessionPayload, emptySessionProfile, type SessionProfileFormValues } from './api/session-payload'
import type { CharacterDetail, CharacterInput, CharacterSummary, CharacterUpdateInput, ChatMessage, CreateSessionInput, SessionMeta, SessionState } from './api/types'
import { loadActiveSessionId, saveActiveSessionId } from './chat/active-session'
import { mergeOpening, reconcileMessages } from './chat/chat-events'
import { applyQuickCommand } from './chat/quick-command'
import { resyncSession, useChatSocket } from './chat/use-chat-socket'
import { formatMessageTime } from './utils/format'
import { formatClothing } from './utils/state-format'
import './App.css'

const relationLabels: Record<string, string> = { step_sister: '妹妹', classmate: '同学', childhood_friend: '青梅竹马', senior: '学姐', teacher: '老师', neighbor: '邻居', other: '其他' }
const clothingLabels: Record<string, string> = { casual: '便装', school_uniform: '校服', pajamas: '睡衣', bath_towel: '浴巾', underwear: '内衣', nude: '未着衣物', dressed: '穿着整齐' }
const clothingOptions = Object.entries(clothingLabels)
const emotionLabels: Record<string, string> = { calm: '平静', happy: '开心', happiness: '开心', joy: '喜悦', excited: '兴奋', excitement: '兴奋', shy: '害羞', shyness: '害羞', shame: '羞耻', embarrassment: '尴尬', nervous: '紧张', anxious: '不安', anxiety: '不安', sad: '难过', hurt: '委屈', angry: '生气', frustration: '烦闷', fear: '害怕', shock: '震惊', surprised: '惊讶', surprise: '惊讶', love: '心动', pleasure: '愉悦', jealousy: '吃醋', anticipation: '期待', satisfaction: '满足', overwhelm: '不知所措', loss_of_control: '失控', inevitability: '难以抗拒', pain: '疼痛', sleepy: '困倦', longing: '渴望', playfulness: '俏皮', trust: '信任', neutral: '平静' }
const localizeEmotion = (value?: string) => value ? value.split(/([/+])/).map((part) => part === '/' || part === '+' ? part : emotionLabels[part.trim().toLowerCase()] || '未知情绪').join('') : '平静'
const phaseLabels: Record<string, string> = { baseline: '平静 / 基线', excitement: '兴奋期', plateau: '平台期', orgasm: '高潮期', resolution: '消退期' }
const stageLabels: Record<string, string> = { A: '日常学习', B: '同学校园', C: '家庭', D: '日常亲密', E: '感情互动', F: '暧昧边缘', G: '初阶性接触', H: '口交', I: '性爱', J: '节日特殊', K: '争吵小情绪' }
const emptyCharacter: CharacterInput = { name: '', age: 18, adult_confirmed: false, relationship_type: 'other', character_description: '', backstory: '', initial_outfit: 'casual', initial_closeness: 0.3, initial_trust: 0.3, likes: [], dislikes: [], fears: [] }

const sessionOutfitPreview = (
  mode: 'automatic' | 'preset' | 'custom',
  value: string,
  templateOutfit?: string,
) => {
  const selected = value.trim()
  if (mode === 'custom') return selected || '等待填写自定义衣着'
  if (mode === 'preset') return selected ? clothingLabels[selected] || selected : '等待选择预设衣着'
  return templateOutfit ? `自动决定（角色模板：${clothingLabels[templateOutfit] || templateOutfit}）` : '自动生成'
}

function Avatar({ character, size = 'normal' }: { character: CharacterSummary; size?: 'normal' | 'small' | 'large' }) {
  const [failed, setFailed] = useState(false)
  const source = character.avatar ? characterMediaUrl(character.id, 'avatar', character.avatar_revision) : ''
  if (!source || failed) return <span className={`avatar avatar--${size} avatar--fallback`}>{character.name.slice(0, 1)}</span>
  return <img className={`avatar avatar--${size}`} src={source} alt="" onError={() => setFailed(true)} />
}

function StatusDot({ tone, label }: { tone: 'ready' | 'busy' | 'offline'; label: string }) {
  return <span className={`status-badge status-badge--${tone}`}><span aria-hidden="true" />{label}</span>
}

function MessageBubble({ message, character, retryDisabled, onRetry }: { message: ChatMessage; character?: CharacterSummary; retryDisabled: boolean; onRetry: (turn: number) => void }) {
  const mine = message.role === 'user'
  const retriable = message.role === 'assistant' && !message.pending && Number.isInteger(message.turn) && message.turn! > 0
  if (message.role === 'system' || message.role === 'narrator') return <div className="system-message">{message.content}</div>
  return <article className={`message ${mine ? 'message--mine' : ''} ${message.pending ? 'message--pending' : ''}`}>
    {!mine && character ? <Avatar character={character} size="small" /> : null}
    <div className="message__body"><div className="message__meta"><span>{mine ? '我' : character?.name || '角色'}</span><time>{formatMessageTime(message.timestamp)}</time></div><div className="message__bubble">{message.content}</div>{retriable && <div className="message__actions"><button type="button" disabled={retryDisabled} onClick={() => onRetry(message.turn!)} aria-label={`重说第 ${message.turn} 回合`}>重说</button></div>}</div>
  </article>
}

function Metric({ label, value, kind = 'normal' }: { label: string; value: number; kind?: 'normal' | 'warm' }) {
  const normalized = Math.max(0, Math.min(1, value))
  return <div className="metric"><div className="metric__label"><span>{label}</span><strong>{Math.round(normalized * 100)}%</strong></div><div className="metric__track" role="meter" aria-label={label} aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(normalized * 100)}><span className={kind === 'warm' ? 'metric__fill metric__fill--warm' : 'metric__fill'} style={{ width: `${normalized * 100}%` }} /></div></div>
}

function ObjectivePanel({ state, session }: { state?: SessionState; session?: SessionMeta }) {
  if (!state) return <div className="panel-empty">开始会话后，这里会显示角色与场景状态。</div>
  const privacyLevel = state.privacy_level ?? (state.privacy === '私密' ? 0.8 : 0.4)
  const dangerLevel = state.danger_level ?? 0
  const detectionRisk = state.detection_risk ?? (state.privacy === '危险' ? 0.8 : 0)
  const hasPeople = state.has_people ?? state.privacy === '危险'
  const othersPresent = state.others_present ?? hasPeople
  const privacyRisk = detectionRisk > 0.5 || state.privacy === '危险' ? '场景中有人或容易被发现，互动可能被打断。' : privacyLevel < 0.6 ? '场景并非完全私密，请留意他人接近或意外打断。' : '场景较为私密，被他人发现或打断的风险较低。'
  const relationshipStage = state.relationship_stage || state.stage
  const scenarioStage = state.scenario_stage || (['J', 'K'].includes(state.stage) ? state.stage : undefined)
  const relationshipLabel = stageLabels[relationshipStage] || `当前阶段：${relationshipStage || '未知'}`
  const scenarioLabel = scenarioStage ? `剧情 ${scenarioStage}：${stageLabels[scenarioStage] || '特殊事件'}` : ''
  return <div className="objective-content">
    <section className="state-section"><div className="section-title"><span>关系状态</span><span className="stage-chip">{relationshipLabel}</span></div>{scenarioLabel && <p className="phase-line">{scenarioLabel}</p>}<Metric label="信任" value={state.trust} /><Metric label="亲密" value={state.closeness} /><Metric label="接受度" value={state.acceptance} /></section>
    <section className="state-section"><div className="section-title"><span>身心状态</span><HeartPulse size={17} aria-hidden="true" /></div><div className="vitals"><div><span>心率</span><strong>{state.heart_rate}</strong><small>BPM</small></div><div><span>情绪</span><strong className="vitals__text">{localizeEmotion(state.emotion)}</strong></div></div><Metric label="体力" value={state.stamina} /><Metric label="疲劳" value={state.fatigue} kind="warm" /><Metric label="唤起" value={state.arousal} kind="warm" /><p className="phase-line">生理阶段：{state.arousal <= 0 ? phaseLabels.baseline : phaseLabels[state.phase] || state.phase}</p></section>
    <section className="state-section"><div className="section-title"><span>场景与风险</span><Activity size={17} aria-hidden="true" /></div><dl className="scene-list"><div><dt>地点</dt><dd>{state.location || session?.current_location || '未知'}</dd></div><div><dt>时间</dt><dd>{state.time_str || session?.current_time || '未知'}</dd></div><div><dt>私密度</dt><dd>{state.privacy}（{Math.round(privacyLevel * 100)}%）</dd></div><div><dt>危险度</dt><dd>{Math.round(dangerLevel * 100)}%</dd></div><div><dt>发现风险</dt><dd>{Math.round(detectionRisk * 100)}%</dd></div><div><dt>在场人物</dt><dd>{hasPeople ? (othersPresent ? '有其他人物' : '有人在场') : '无其他人物'}</dd></div><div><dt>衣着</dt><dd>{formatClothing(state.clothing)}</dd></div></dl><p className={`privacy-note privacy-note--${state.privacy === '危险' ? 'risk' : 'safe'}`}><CircleAlert size={15} />{privacyRisk}</p></section>
    {(state.is_interrupted || state.is_orgasm) && <div className="state-alert"><CircleAlert size={16} />{state.is_interrupted ? '本回合发生中断' : '本回合存在显著身体反应'}</div>}
  </div>
}

function CharacterForm({ initial, busy, deleting, onCancel, onDelete, onSave }: { initial?: CharacterDetail; busy: boolean; deleting?: boolean; onCancel: () => void; onDelete?: () => void; onSave: (value: CharacterInput) => void }) {
  const editing = Boolean(initial)
  const [value, setValue] = useState<CharacterInput>(() => initial ? { name: initial.name, age: initial.age ?? 0, adult_confirmed: initial.adult_verified && (initial.age ?? 0) >= 18, relationship_type: initial.relationship_type, character_description: initial.character_description, backstory: initial.backstory, initial_outfit: initial.initial_outfit, initial_closeness: initial.initial_closeness, initial_trust: initial.initial_trust, likes: initial.likes, dislikes: initial.dislikes, fears: initial.fears } : emptyCharacter)
  const textList = (items?: string[]) => items?.join('、') || ''
  const patch = <K extends keyof CharacterInput>(key: K, next: CharacterInput[K]) => setValue((current) => ({ ...current, [key]: next }))
  const patchAge = (age: number) => setValue((current) => ({ ...current, age, adult_confirmed: age >= 18 && current.adult_confirmed }))
  return <form className="character-form" onSubmit={(event) => { event.preventDefault(); onSave(value) }}>
    <div className="form-grid"><label><span>名字</span><input required maxLength={40} value={value.name} onChange={(e) => patch('name', e.target.value)} /></label><label><span>年龄</span><input required min={0} max={120} type="number" value={value.age} onChange={(e) => patchAge(Number(e.target.value))} /></label></div>
    <label><span>关系{editing && <small>创建后不可修改</small>}</span><select value={value.relationship_type} disabled={editing} aria-describedby={editing ? 'relationship-immutable-note' : undefined} onChange={(e) => patch('relationship_type', e.target.value)}>{Object.entries(relationLabels).map(([id, label]) => <option key={id} value={id}>{label}</option>)}</select>{editing && <small id="relationship-immutable-note" className="immutable-note">关系类型由会话数据引用，编辑角色时不可修改。</small>}</label>
    <label><span>角色描述</span><textarea required rows={3} value={value.character_description} onChange={(e) => patch('character_description', e.target.value)} /></label>
    <label><span>背景故事</span><textarea rows={3} value={value.backstory} onChange={(e) => patch('backstory', e.target.value)} /></label>
    <label><span>初始衣着（角色模板默认值）</span><select value={value.initial_outfit} onChange={(e) => patch('initial_outfit', e.target.value)}>{clothingOptions.map(([id, label]) => <option key={id} value={id}>{label}</option>)}</select></label>
    <div className="form-grid"><label><span>初始信任（角色模板默认值）{editing && <small>创建后不可修改</small>}</span><input type="number" min={0} max={1} step={0.05} value={value.initial_trust} disabled={editing} onChange={(e) => patch('initial_trust', Number(e.target.value))} /></label><label><span>初始亲密（角色模板默认值）{editing && <small>创建后不可修改</small>}</span><input type="number" min={0} max={1} step={0.05} value={value.initial_closeness} disabled={editing} onChange={(e) => patch('initial_closeness', Number(e.target.value))} /></label></div>
    {editing && <p className="immutable-note immutable-note--block">初始信任与初始亲密仅用于角色创建，现有角色不能在此修改。</p>}
    <div className="form-grid"><label><span>喜欢（顿号分隔）</span><input value={textList(value.likes)} onChange={(e) => patch('likes', e.target.value.split(/[、,，]/).map((x) => x.trim()).filter(Boolean))} /></label><label><span>不喜欢（顿号分隔）</span><input value={textList(value.dislikes)} onChange={(e) => patch('dislikes', e.target.value.split(/[、,，]/).map((x) => x.trim()).filter(Boolean))} /></label></div>
    <label><span>害怕（顿号分隔）</span><input value={textList(value.fears)} onChange={(e) => patch('fears', e.target.value.split(/[、,，]/).map((x) => x.trim()).filter(Boolean))} /></label>
    <label className="check-row"><input type="checkbox" checked={value.adult_confirmed} disabled={value.age < 18} onChange={(e) => patch('adult_confirmed', e.target.checked)} /><span>确认该角色当前已年满 18 岁；成人互动要求年龄至少 18 岁且完成此确认</span></label>
    <p className="immutable-note immutable-note--block">年龄缺失、未满 18 岁或未完成成年人确认时，历史仍可查看，但成人互动与成人剧情会暂停。</p>
    <div className={`form-actions ${onDelete ? 'form-actions--split' : ''}`}>{onDelete && <button type="button" className="button button--danger" disabled={deleting || busy} onClick={onDelete}><Trash2 size={16} />{deleting ? '删除中…' : '删除角色'}</button>}<span className="form-actions__end"><button type="button" className="button button--ghost" onClick={onCancel}>取消</button><button className="button button--primary" disabled={busy || deleting}>{busy ? '保存中…' : '保存角色'}</button></span></div>
  </form>
}

function App() {
  const queryClient = useQueryClient()
  const [selectedSessionId, setSelectedSessionId] = useState<string | undefined>(() => loadActiveSessionId())
  const [configCharacterId, setConfigCharacterId] = useState<string>()
  const [expanded, setExpanded] = useState<Set<string>>(new Set())
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [objectiveState, setObjectiveState] = useState<SessionState>()
  const messageCacheRef = useRef(new Map<string, ChatMessage[]>())
  const stateCacheRef = useRef(new Map<string, SessionState>())
  const [draft, setDraft] = useState('')
  const [search, setSearch] = useState('')
  const [sessionName, setSessionName] = useState('')
  const [scenarioId, setScenarioId] = useState('')
  const [customOpening, setCustomOpening] = useState('')
  const [sessionOverrides, setSessionOverrides] = useState({ initial_trust: '', initial_closeness: '', outfit_mode: 'automatic' as 'automatic' | 'preset' | 'custom', outfit: '', location_mode: 'automatic' as 'automatic' | 'custom', location: '', time_str: '', privacy: '' as '' | NonNullable<CreateSessionInput['privacy']> })
  const [sessionProfile, setSessionProfile] = useState<SessionProfileFormValues>(emptySessionProfile)
  const [commandOpen, setCommandOpen] = useState(false)
  const [editorId, setEditorId] = useState<string | 'new'>()
  const [leftOpen, setLeftOpen] = useState(false)
  const [rightOpen, setRightOpen] = useState(false)
  const messageEndRef = useRef<HTMLDivElement>(null)
  const composerTextareaRef = useRef<HTMLTextAreaElement>(null)
  const selectedSessionIdRef = useRef(selectedSessionId)
  selectedSessionIdRef.current = selectedSessionId

  const healthQuery = useQuery({ queryKey: ['health'], queryFn: ({ signal }) => appApi.health(signal), retry: 1, refetchInterval: 15_000 })
  const modelQuery = useQuery({ queryKey: ['model-status'], queryFn: ({ signal }) => appApi.modelStatus(signal), retry: 1, refetchInterval: (query) => query.state.data?.loading ? 2_000 : 15_000 })
  const menuQuery = useQuery({ queryKey: ['menu'], queryFn: ({ signal }) => appApi.menu(signal), retry: 1 })
  const commandQuery = useQuery({ queryKey: ['commands', selectedSessionId], queryFn: ({ signal }) => sessionsApi.commands(selectedSessionId!, signal), enabled: Boolean(selectedSessionId && commandOpen), staleTime: 300_000 })
  const sessionQuery = useQuery({ queryKey: ['session', selectedSessionId], queryFn: ({ signal }) => sessionsApi.detail(selectedSessionId!, signal), enabled: Boolean(selectedSessionId) })
  const previewQuery = useQuery({ queryKey: ['character', configCharacterId], queryFn: ({ signal }) => charactersApi.detail(configCharacterId!, signal), enabled: Boolean(configCharacterId) })
  const editorQuery = useQuery({ queryKey: ['character', editorId], queryFn: ({ signal }) => charactersApi.detail(editorId!, signal), enabled: Boolean(editorId && editorId !== 'new') })

  const characters = useMemo(() => menuQuery.data?.characters || [], [menuQuery.data?.characters])
  const sessions = useMemo(() => menuQuery.data?.sessions || [], [menuQuery.data?.sessions])
  const selectedSession = sessionQuery.data?.meta || sessions.find((session) => session.session_id === selectedSessionId)
  const selectedCharacter = characters.find((character) => character.id === selectedSession?.character_id)
  const configCharacter = characters.find((character) => character.id === configCharacterId)

  useEffect(() => {
    if (!menuQuery.data || !selectedSessionId) return
    if (sessions.some((session) => session.session_id === selectedSessionId)) return
    saveActiveSessionId()
    setSelectedSessionId(undefined)
    setMessages([])
    setObjectiveState(undefined)
  }, [menuQuery.data, selectedSessionId, sessions])

  useEffect(() => {
    if (!selectedSessionId || !selectedCharacter) return
    setExpanded((current) => {
      if (current.has(selectedCharacter.id)) return current
      const next = new Set(current)
      next.add(selectedCharacter.id)
      return next
    })
  }, [selectedCharacter, selectedSessionId])

  useEffect(() => {
    if (!selectedSessionId || !sessionQuery.data) return
    setMessages((current) => {
      const reconciled = reconcileMessages(current, sessionQuery.data.messages)
      messageCacheRef.current.set(selectedSessionId, reconciled)
      return reconciled
    })
    stateCacheRef.current.set(selectedSessionId, sessionQuery.data.state)
    setObjectiveState(sessionQuery.data.state)
  }, [selectedSessionId, sessionQuery.data])
  const handleMessages = useCallback<Dispatch<SetStateAction<ChatMessage[]>>>((update) => {
    const id = selectedSessionIdRef.current
    if (!id) return
    setMessages((current) => {
      const next = typeof update === 'function' ? update(current) : update
      messageCacheRef.current.set(id, next)
      return next
    })
  }, [])
  const handleResync = useCallback(async (targetSessionId?: string) => {
    const id = targetSessionId || selectedSessionIdRef.current
    if (!id) return
    const synced = await resyncSession(id)
    queryClient.setQueryData(['session', id], synced.detail)
    const cached = messageCacheRef.current.get(id) || []
    messageCacheRef.current.set(id, reconcileMessages(cached, synced.history.messages))
    stateCacheRef.current.set(id, synced.state.state)
    if (selectedSessionIdRef.current === id) {
      setMessages(messageCacheRef.current.get(id) || [])
      setObjectiveState(synced.state.state)
    }
    await queryClient.invalidateQueries({ queryKey: ['menu'] })
  }, [queryClient])
  const { chat, send, retry, stop } = useChatSocket({ sessionId: selectedSessionId, onMessages: handleMessages, onResync: handleResync })
  useEffect(() => {
    if (!selectedSessionId || !chat.state) return
    stateCacheRef.current.set(selectedSessionId, chat.state)
    setObjectiveState(chat.state)
  }, [chat.state, selectedSessionId])
  useEffect(() => { messageEndRef.current?.scrollIntoView({ block: 'end' }) }, [messages, chat.generation])
  useEffect(() => { if (!selectedSessionId) return; const save = () => void sessionsApi.autosave(selectedSessionId).catch(() => undefined); const timer = window.setInterval(save, 60_000); return () => window.clearInterval(timer) }, [selectedSessionId])

  const createSession = useMutation({ mutationFn: () => sessionsApi.create(buildCreateSessionPayload({ characterId: configCharacterId!, name: sessionName, scenarioId, customOpening, initialTrust: sessionOverrides.initial_trust, initialCloseness: sessionOverrides.initial_closeness, outfit: { mode: sessionOverrides.outfit_mode, value: sessionOverrides.outfit }, location: { mode: sessionOverrides.location_mode, value: sessionOverrides.location }, timeStr: sessionOverrides.time_str, privacy: sessionOverrides.privacy, userProfile: sessionProfile })), onSuccess: async (data) => { const opening = data.opening_message ? [{ ...data.opening_message, pending: false }] : mergeOpening([], data.opening); messageCacheRef.current.set(data.session_id, opening); stateCacheRef.current.set(data.session_id, data.state); setMessages(opening); setObjectiveState(data.state); saveActiveSessionId(data.session_id); setSelectedSessionId(data.session_id); setConfigCharacterId(undefined); setLeftOpen(false); queryClient.setQueryData(['session', data.session_id], { meta: data.meta, messages: opening, state: data.state, scenario_id: data.scenario_id, snapshot_id: data.snapshot_id, timeline_id: data.timeline_id, user_profile: data.user_profile }); await queryClient.invalidateQueries({ queryKey: ['menu'] }) } })
  const deleteSession = useMutation({ mutationFn: sessionsApi.delete, onSuccess: async (_, id) => { if (selectedSessionId === id) { saveActiveSessionId(); setSelectedSessionId(undefined); setMessages([]); setObjectiveState(undefined) } await queryClient.invalidateQueries({ queryKey: ['menu'] }) } })
  const saveCharacter = useMutation({ mutationFn: (value: CharacterInput) => { if (editorId === 'new') return charactersApi.create(value); const update: CharacterUpdateInput = { name: value.name, age: value.age, adult_verified: value.adult_confirmed, character_description: value.character_description, backstory: value.backstory, initial_outfit: value.initial_outfit, likes: value.likes, dislikes: value.dislikes, fears: value.fears }; return charactersApi.update(editorId!, update) }, onSuccess: async () => { setEditorId(undefined); await queryClient.invalidateQueries({ queryKey: ['menu'] }) } })
  const deleteCharacter = useMutation({ mutationFn: charactersApi.delete, onSuccess: async (_, id) => { setEditorId(undefined); if (configCharacterId === id) setConfigCharacterId(undefined); if (selectedCharacter?.id === id) { saveActiveSessionId(); setSelectedSessionId(undefined); setMessages([]); setObjectiveState(undefined) } await queryClient.invalidateQueries({ queryKey: ['menu'] }) } })

  const grouped = useMemo(() => characters.map((character) => ({ character, sessions: sessions.filter((session) => session.character_id === character.id).sort((a, b) => Date.parse(b.last_active_at) - Date.parse(a.last_active_at)) })).filter(({ character }) => { const value = search.trim().toLocaleLowerCase(); return !value || character.name.toLocaleLowerCase().includes(value) || character.description_preview.toLocaleLowerCase().includes(value) }), [characters, search, sessions])
  const commandGroups = useMemo(() => { const groupLabels: Record<string, string> = { flow: '流程', time: '时间', location: '地点', daily: '日常动作', intimacy: '亲密动作' }; const groups = new Map<string, NonNullable<typeof commandQuery.data>>(); for (const command of commandQuery.data || []) { const key = command.group?.trim() || '其他'; const group = groupLabels[key] || key; groups.set(group, [...(groups.get(group) || []), command]) } return [...groups.entries()] }, [commandQuery.data])
  const canSendContent = Boolean(selectedSessionId && chat.generation === 'idle' && modelQuery.data?.main_model_loaded)
  const retryDisabled = !canSendContent
  const canSend = Boolean(draft.trim() && canSendContent)
  const chooseSession = (id: string) => {
    saveActiveSessionId(id)
    setSelectedSessionId(id)
    setConfigCharacterId(undefined)
    setMessages(messageCacheRef.current.get(id) || [])
    setObjectiveState(stateCacheRef.current.get(id))
    setLeftOpen(false)
  }
  const beginNew = (character: CharacterSummary) => { saveActiveSessionId(); setConfigCharacterId(character.id); setSelectedSessionId(undefined); setMessages([]); setObjectiveState(undefined); setSessionName(`与${character.name}的新会话`); setScenarioId(''); setCustomOpening(''); setSessionOverrides({ initial_trust: '', initial_closeness: '', outfit_mode: 'automatic', outfit: '', location_mode: 'automatic', location: '', time_str: '', privacy: '' }); setSessionProfile(emptySessionProfile()); setLeftOpen(false) }
  const chooseCommand = (content: string) => applyQuickCommand(content, { setDraft, close: () => setCommandOpen(false), focus: () => window.requestAnimationFrame(() => composerTextareaRef.current?.focus()) })
  const sendContent = (content: string) => { const trimmed = content.trim(); if (!trimmed || !canSendContent || !send(trimmed)) return false; setCommandOpen(false); return true }
  const submit = (event: FormEvent) => { event.preventDefault(); if (sendContent(draft)) setDraft('') }
  const backendOffline = healthQuery.isError
  const model = modelQuery.data
  const connectionLabel = chat.generation === 'committing' ? '正在提交' : chat.generation !== 'idle' ? '模型生成中' : backendOffline ? '后端离线' : model?.main_model_loaded ? '模型就绪' : model?.loading ? '模型加载中' : '模型未就绪'
  const connectionTone = chat.generation !== 'idle' ? 'busy' : backendOffline || !model?.main_model_loaded ? 'offline' : 'ready'

  if (menuQuery.isLoading && !menuQuery.data) return <main className="boot-screen"><RefreshCw className="spin" /><p>正在连接本地服务并载入会话...</p></main>
  if (menuQuery.isError && !menuQuery.data) { const message = menuQuery.error instanceof ApiError ? menuQuery.error.message : '无法载入应用数据'; return <main className="boot-screen boot-screen--error"><CircleAlert /><h1>后端连接失败</h1><p>{message}</p><button onClick={() => void menuQuery.refetch()}><RefreshCw size={17} />重新连接</button></main> }

  return <main className={`app-shell ${!selectedSessionId && !configCharacterId ? 'app-shell--welcome' : ''}`}>
    <button className="mobile-fab mobile-fab--left" aria-label="打开角色列表" onClick={() => setLeftOpen(true)}><Menu /></button><button className="mobile-fab mobile-fab--right" aria-label="打开状态面板" onClick={() => setRightOpen(true)}><PanelRight /></button>
    {(leftOpen || rightOpen) && <button className="drawer-backdrop" aria-label="关闭面板" onClick={() => { setLeftOpen(false); setRightOpen(false) }} />}
    <aside className={`character-pane ${leftOpen ? 'is-open' : ''}`}><header className="brand-row"><div><span className="brand-mark"><Heart size={18} fill="currentColor" /></span><strong>Heart Chat</strong></div><div className="brand-actions"><button className="icon-button" aria-label="创建自定义角色" title="创建自定义角色" onClick={() => setEditorId('new')}><Plus /></button><button className="icon-button drawer-close" aria-label="关闭角色列表" onClick={() => setLeftOpen(false)}><X /></button></div></header>
      <label className="search-box"><Search size={17} aria-hidden="true" /><input name="character-search" autoComplete="off" value={search} onChange={(e) => setSearch(e.target.value)} placeholder="搜索角色" aria-label="搜索角色" /></label>
      <div className="character-list">{grouped.map(({ character, sessions: history }) => { const isExpanded = expanded.has(character.id); return <div className={`character-entry ${selectedCharacter?.id === character.id || configCharacterId === character.id ? 'is-active' : ''}`} key={character.id} onClick={() => setExpanded((current) => { const next = new Set(current); if (next.has(character.id)) next.delete(character.id); else next.add(character.id); return next })}>
        <div className="character-main"><button className="character-toggle" aria-expanded={isExpanded} onClick={(event) => { event.stopPropagation(); setExpanded((current) => { const next = new Set(current); if (next.has(character.id)) next.delete(character.id); else next.add(character.id); return next }) }}>{isExpanded ? <ChevronDown /> : <ChevronRight />}<Avatar character={character} /><span className="character-copy"><span className="character-name"><strong>{character.name}</strong><small>{relationLabels[character.relationship_type] || character.relationship_type}</small></span><span className="character-preview">{character.description_preview || '等待与你相遇'}</span></span></button>{!character.is_preset && <button className="mini-action" aria-label={`编辑${character.name}`} title="编辑角色" onClick={(event) => { event.stopPropagation(); setEditorId(character.id) }}><Edit3 /></button>}</div>
        <button className="new-chat-action" onClick={(event) => { event.stopPropagation(); beginNew(character) }}><MessageSquarePlus />新建聊天</button>
        {isExpanded && <div className="session-list">{history.map((session) => <div className="session-row" key={session.session_id}><button className={session.session_id === selectedSessionId ? 'is-current session-choice' : 'session-choice'} onClick={(event) => { event.stopPropagation(); chooseSession(session.session_id) }}><span><strong>{session.session_name}</strong><small><time>{new Date(session.created_at).toLocaleString('zh-CN')}</time> · {session.total_interactions} 回合</small></span></button><button className="delete-action" aria-label={`删除会话${session.session_name}`} onClick={(event) => { event.stopPropagation(); if (window.confirm(`确定删除“${session.session_name}”吗？此操作无法撤销。`)) deleteSession.mutate(session.session_id) }}><Trash2 /></button></div>)}</div>}
      </div>})}{!grouped.length && <div className="panel-empty">没有匹配的角色</div>}</div>
    </aside>

    <section className={`chat-pane ${!selectedSessionId ? 'chat-pane--landing' : ''}`}><header className="chat-header"><div className="chat-identity">{selectedCharacter ? <Avatar character={selectedCharacter} size="small" /> : configCharacter ? <Avatar character={configCharacter} size="small" /> : null}<div><h1>{selectedSession?.session_name || (configCharacter ? `与 ${configCharacter.name} 开始聊天` : 'Heart Chat')}</h1><p>{selectedCharacter?.name}{selectedSession ? ` · ${selectedSession.current_location}` : configCharacter ? '配置新会话' : '本地、私密、由你掌控'}</p></div></div>{selectedSessionId && <div className="header-status"><StatusDot tone={connectionTone} label={connectionLabel} /></div>}</header>
      {backendOffline && <div className="blocking-banner"><CircleAlert size={17} />后端连接中断，历史仍可查看，发送已暂停。</div>}
      {!selectedSessionId ? <div className={`landing-content ${configCharacter ? 'landing-content--config' : 'landing-content--welcome'}`}>{configCharacter ? <section className="session-config"><span className="eyebrow">NEW CONVERSATION</span><h2>准备好与 {configCharacter.name} 相遇了吗？</h2><p>给这次聊天起个名字。留空开场白会自动生成，也可以由你亲自写下开场。</p><div className="override-section override-section--template"><div className="override-heading"><strong>角色长期模板</strong><small>这里展示角色创建时保存的默认值；它不会因本次会话覆盖而被修改。</small></div><dl className="template-summary"><div><dt>关系</dt><dd>{relationLabels[configCharacter.relationship_type] || configCharacter.relationship_type}</dd></div><div><dt>默认信任</dt><dd>{previewQuery.data ? Math.round(previewQuery.data.initial_trust * 100) + '%' : '载入中…'}</dd></div><div><dt>默认亲密</dt><dd>{previewQuery.data ? Math.round(previewQuery.data.initial_closeness * 100) + '%' : '载入中…'}</dd></div><div><dt>默认衣着</dt><dd>{previewQuery.data ? clothingLabels[previewQuery.data.initial_outfit] || previewQuery.data.initial_outfit : '载入中…'}</dd></div></dl><small className="inheritance-note">生效优先级：角色模板 → 剧情卡 → 本次显式覆盖。</small></div><label><span>会话名称</span><input maxLength={80} placeholder={`与${configCharacter.name}的新会话`} value={sessionName} onChange={(e) => setSessionName(e.target.value)} /></label><label><span>自定义开场白（可选）</span><textarea rows={3} maxLength={2000} placeholder="例如：雨夜回到家，发现她正在客厅等你……" value={customOpening} onChange={(e) => { setCustomOpening(e.target.value); if (e.target.value.trim()) setScenarioId('') }} /></label><p className="immutable-note">不填写时由 Heart Chat 根据剧情与最终状态自动生成合适的开场。</p><div className="override-section override-section--session"><div className="override-heading"><strong>本次开场覆盖</strong><small>“自动生成”不会向后端发送该字段，由角色模板与剧情卡决定；只有明确填写的项目才会覆盖。</small></div><div className="override-fields"><div className="form-grid"><label><span>初始信任</span><input aria-label="初始信任" type="number" min={0} max={1} step={0.05} placeholder="自动生成" value={sessionOverrides.initial_trust} onChange={(e) => setSessionOverrides((current) => ({ ...current, initial_trust: e.target.value }))} /></label><label><span>初始亲密</span><input aria-label="初始亲密" type="number" min={0} max={1} step={0.05} placeholder="自动生成" value={sessionOverrides.initial_closeness} onChange={(e) => setSessionOverrides((current) => ({ ...current, initial_closeness: e.target.value }))} /></label></div><label><span>初始衣着</span><select aria-label="初始衣着生成方式" value={sessionOverrides.outfit_mode} onChange={(e) => setSessionOverrides((current) => ({ ...current, outfit_mode: e.target.value as 'automatic' | 'preset' | 'custom', outfit: '' }))}><option value="automatic">自动生成</option><option value="preset">预设衣着</option><option value="custom">自定义描述</option></select></label>{sessionOverrides.outfit_mode === 'preset' && <label><span>预设衣着</span><select aria-label="预设衣着" value={sessionOverrides.outfit} onChange={(e) => setSessionOverrides((current) => ({ ...current, outfit: e.target.value }))}><option value="">请选择</option>{clothingOptions.map(([id, label]) => <option key={id} value={id}>{label}</option>)}</select></label>}{sessionOverrides.outfit_mode === 'custom' && <label><span>自定义衣着描述</span><input aria-label="自定义衣着描述" placeholder="例如：宽松的白色针织衫和深色长裙" value={sessionOverrides.outfit} onChange={(e) => setSessionOverrides((current) => ({ ...current, outfit: e.target.value }))} /></label>}<label><span>地点</span><select aria-label="开场地点生成方式" value={sessionOverrides.location_mode} onChange={(e) => setSessionOverrides((current) => ({ ...current, location_mode: e.target.value as 'automatic' | 'custom' }))}><option value="automatic">自动生成</option><option value="custom">自定义</option></select></label>{sessionOverrides.location_mode === 'custom' && <label><span>自定义地点描述</span><input aria-label="自定义地点描述" placeholder="例如：雨夜里安静的客厅" value={sessionOverrides.location} onChange={(e) => setSessionOverrides((current) => ({ ...current, location: e.target.value }))} /></label>}<div className="form-grid"><label><span>时间 <small>如 2026年8月9日06：30 或 第2天下午3点30分</small></span><input aria-label="开场时间" placeholder="自动生成" value={sessionOverrides.time_str} onChange={(e) => setSessionOverrides((current) => ({ ...current, time_str: e.target.value }))} /></label><label><span>私密度</span><select aria-label="开场私密度" value={sessionOverrides.privacy} onChange={(e) => setSessionOverrides((current) => ({ ...current, privacy: e.target.value as '' | NonNullable<CreateSessionInput['privacy']> }))}><option value="">自动生成</option><option value="私密">私密</option><option value="半公开">半公开</option><option value="危险">危险</option></select></label></div></div></div><div className="override-section override-section--profile"><div className="override-heading"><strong>用户在本会话中的身份</strong><small>全部留空时逐字段继承当前全局画像；填写的项目标记为本会话覆盖，并随存档与分支保存。</small></div><p className="inheritance-note">当前状态：{Object.values(sessionProfile).some((value) => value.trim()) ? '包含本会话覆盖；其余字段继承全局画像' : '全部继承全局画像'}</p><div className="override-fields"><div className="form-grid"><label><span>姓名</span><input maxLength={80} placeholder="继承全局画像" value={sessionProfile.name} onChange={(e) => setSessionProfile((current) => ({ ...current, name: e.target.value }))} /></label><label><span>年龄</span><input type="number" min={0} max={120} placeholder="继承全局画像" value={sessionProfile.age} onChange={(e) => setSessionProfile((current) => ({ ...current, age: e.target.value }))} /></label></div><div className="form-grid"><label><span>性别</span><input maxLength={40} placeholder="继承全局画像" value={sessionProfile.gender} onChange={(e) => setSessionProfile((current) => ({ ...current, gender: e.target.value }))} /></label><label><span>希望角色如何称呼你</span><input maxLength={80} placeholder="继承该角色的称呼偏好" value={sessionProfile.preferredAddress} onChange={(e) => setSessionProfile((current) => ({ ...current, preferredAddress: e.target.value }))} /></label></div><label><span>与角色的关系</span><input maxLength={120} placeholder="继承全局画像中的角色关系" value={sessionProfile.relation} onChange={(e) => setSessionProfile((current) => ({ ...current, relation: e.target.value }))} /></label><label><span>性格描述</span><textarea rows={2} maxLength={1200} placeholder="继承全局画像" value={sessionProfile.personalityDescription} onChange={(e) => setSessionProfile((current) => ({ ...current, personalityDescription: e.target.value }))} /></label><label><span>说话风格</span><input maxLength={500} placeholder="继承全局画像" value={sessionProfile.speakingStyleHint} onChange={(e) => setSessionProfile((current) => ({ ...current, speakingStyleHint: e.target.value }))} /></label><label><span>外观描述</span><textarea rows={2} maxLength={1000} placeholder="继承全局画像" value={sessionProfile.appearanceHint} onChange={(e) => setSessionProfile((current) => ({ ...current, appearanceHint: e.target.value }))} /></label><label><span>特点（顿号或逗号分隔）</span><input placeholder="例如：细心、直接、喜欢安静" value={sessionProfile.traits} onChange={(e) => setSessionProfile((current) => ({ ...current, traits: e.target.value }))} /></label><label><span>补充背景</span><textarea rows={2} maxLength={1200} placeholder="仅作为本会话资料，不会覆盖系统规则" value={sessionProfile.additionalContext} onChange={(e) => setSessionProfile((current) => ({ ...current, additionalContext: e.target.value }))} /></label></div></div>{createSession.isError && <p className="form-error">{createSession.error instanceof ApiError ? createSession.error.message : '创建会话失败'}</p>}<button className="button button--primary button--wide" onClick={() => createSession.mutate()} disabled={createSession.isPending}>{createSession.isPending ? '正在创建…' : '开始 Heart Chat'}<Heart size={17} /></button></section> : <section className="welcome"><span className="welcome-mark"><Heart fill="currentColor" /></span><h2>选择角色，开始 Heart Chat</h2><p>从左侧展开一位角色的历史，或直接新建一段只保存在本地的对话。</p></section>}</div> : <><div className="message-scroll"><div className="message-column">{sessionQuery.isLoading && !messages.length ? <div className="conversation-empty"><RefreshCw className="spin" /><p>正在载入消息...</p></div> : messages.map((message, index) => <MessageBubble key={message.message_id || message.client_id || `${message.role}-${message.timestamp || index}-${index}`} message={message} character={selectedCharacter} retryDisabled={retryDisabled} onRetry={(turn) => { retry(turn) }} />)}{chat.notices.map((notice) => <div className={`process-notice process-notice--${notice.tone}`} key={notice.id}>{notice.text}</div>)}{(chat.generation !== 'idle' || Boolean(chat.streamBuffer && chat.lastError)) && <article className="message message--streaming">{selectedCharacter ? <Avatar character={selectedCharacter} size="small" /> : null}<div className="message__body"><div className="message__meta"><span>{selectedCharacter?.name || '角色'}</span><span>{chat.generation === 'idle' ? '回复已中断' : chat.generation === 'stopping' ? '正在停止' : chat.generation === 'thinking' ? '正在理解' : '正在回复'}</span></div><div className="message__bubble">{chat.streamBuffer || <span className="typing"><i /><i /><i /></span>}</div></div></article>}{chat.lastError && <div className="inline-error"><CircleAlert size={16} />{chat.lastError}</div>}<div ref={messageEndRef} /></div></div>
        <form className="composer" onSubmit={submit}>{commandOpen && <div className="command-panel" role="dialog" aria-label="快捷指令"><div className="command-panel__head"><strong>快捷指令</strong><button type="button" aria-label="关闭快捷指令" onClick={() => setCommandOpen(false)}><X /></button></div>{commandQuery.isLoading ? <div className="command-panel__state"><RefreshCw className="spin" size={16} /><span>载入指令中…</span></div> : commandQuery.isError ? <div className="command-panel__state command-panel__state--error"><CircleAlert size={16} /><span>快捷指令加载失败</span><button type="button" onClick={() => void commandQuery.refetch()}>重试</button></div> : commandGroups.length ? commandGroups.map(([group, commands]) => <section className="command-group" key={group}><h3>{group}</h3>{commands.map((command) => <button type="button" key={command.id} onClick={() => chooseCommand(command.content)}><strong>{command.label}</strong><span>{command.description}</span></button>)}</section>) : <div className="command-panel__state"><span>当前会话暂无快捷指令</span></div>}</div>}<div className="composer__box"><button className="command-button" type="button" aria-label="打开快捷指令" aria-expanded={commandOpen} onClick={() => setCommandOpen((open) => !open)}><Plus /></button><textarea ref={composerTextareaRef} rows={1} maxLength={2000} value={draft} onChange={(e) => setDraft(e.target.value)} onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); event.currentTarget.form?.requestSubmit() } }} placeholder="输入消息，Shift + Enter 换行" disabled={chat.generation !== 'idle'} aria-label="消息内容" />{chat.generation !== 'idle' ? <button className="send-button send-button--stop" type="button" onClick={stop} disabled={chat.generation === 'stopping'} aria-label="停止生成"><Square size={17} fill="currentColor" /></button> : <button className="send-button" type="submit" disabled={!canSend} aria-label="发送消息"><Send size={18} /></button>}</div><div className="composer__hint"><span>{!model?.main_model_loaded ? '主模型就绪后可发送' : chat.generation !== 'idle' ? '模型生成中' : '每分钟自动保存 · 内容仅存于本地'}</span><span>{draft.length}/2000</span></div></form></>}
    </section>

    <aside className={`objective-pane ${rightOpen ? 'is-open' : ''}`}><header className="objective-header"><div><h2>{configCharacter ? '角色预览' : '客观状态'}</h2><p>{configCharacter ? '未明确填写的设定均自动生成' : '以后端提交状态为准'}</p></div><button className="icon-button drawer-close" aria-label="关闭状态面板" onClick={() => setRightOpen(false)}><X /></button></header>{configCharacter ? <div className="character-detail">{previewQuery.isLoading ? <RefreshCw className="spin" /> : <><Avatar character={configCharacter} size="large" /><h3>{configCharacter.name}</h3><p>{previewQuery.data?.character_description || configCharacter.description_preview}</p><dl><div><dt>关系</dt><dd>{relationLabels[configCharacter.relationship_type] || configCharacter.relationship_type}</dd></div><div><dt>初始信任</dt><dd>{sessionOverrides.initial_trust || '自动生成'}</dd></div><div><dt>初始亲密</dt><dd>{sessionOverrides.initial_closeness || '自动生成'}</dd></div><div><dt>初始衣着</dt><dd>{sessionOutfitPreview(sessionOverrides.outfit_mode, sessionOverrides.outfit, previewQuery.data?.initial_outfit)}</dd></div><div><dt>地点</dt><dd>{sessionOverrides.location_mode === 'custom' && sessionOverrides.location.trim() ? sessionOverrides.location.trim() : '自动生成'}</dd></div><div><dt>时间</dt><dd>{sessionOverrides.time_str.trim() || '自动生成'}</dd></div><div><dt>私密度</dt><dd>{sessionOverrides.privacy || '自动生成'}</dd></div></dl></>}</div> : <ObjectivePanel state={objectiveState} session={selectedSession} />}</aside>

    {editorId && <div className="modal-backdrop" role="presentation" onMouseDown={(e) => { if (e.target === e.currentTarget) setEditorId(undefined) }}><section className="modal" role="dialog" aria-modal="true" aria-labelledby="character-editor-title"><header><div><span className="eyebrow">CUSTOM CHARACTER</span><h2 id="character-editor-title">{editorId === 'new' ? '创建自定义角色' : '编辑自定义角色'}</h2></div><button className="icon-button" aria-label="关闭" onClick={() => setEditorId(undefined)}><X /></button></header>{editorId !== 'new' && editorQuery.isLoading ? <div className="panel-empty"><RefreshCw className="spin" /></div> : <CharacterForm key={editorId} initial={editorQuery.data} busy={saveCharacter.isPending} deleting={deleteCharacter.isPending} onCancel={() => setEditorId(undefined)} onDelete={editorId !== 'new' && editorQuery.data && !editorQuery.data.is_preset ? () => { if (window.confirm(`确定删除“${editorQuery.data!.name}”吗？相关选择将被清除，此操作无法撤销。`)) deleteCharacter.mutate(editorQuery.data!.id) } : undefined} onSave={(value) => saveCharacter.mutate(value)} />}{saveCharacter.isError && <p className="form-error modal-error">{saveCharacter.error instanceof ApiError ? saveCharacter.error.message : '角色保存失败'}</p>}</section></div>}
  </main>
}

export default App
