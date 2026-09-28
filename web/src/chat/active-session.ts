const ACTIVE_SESSION_KEY = 'heart-chat.active-session-id'

export function loadActiveSessionId() {
  try {
    return window.localStorage.getItem(ACTIVE_SESSION_KEY) || undefined
  } catch {
    return undefined
  }
}

export function saveActiveSessionId(sessionId?: string) {
  try {
    if (sessionId) window.localStorage.setItem(ACTIVE_SESSION_KEY, sessionId)
    else window.localStorage.removeItem(ACTIVE_SESSION_KEY)
  } catch {
    // Selection persistence is best-effort; the app remains usable without storage.
  }
}
