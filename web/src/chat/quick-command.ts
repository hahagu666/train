export type QuickCommandActions = {
  setDraft: (content: string) => void
  close: () => void
  focus: () => void
}

export function applyQuickCommand(content: string, actions: QuickCommandActions) {
  actions.setDraft(content)
  actions.close()
  actions.focus()
}
