const clothingLabels: Record<string, string> = {
  casual: '便装',
  school_uniform: '校服',
  pajamas: '睡衣',
  bath_towel: '浴巾',
  underwear: '内衣',
  nude: '未着衣物',
  dressed: '穿着整齐',
}

export function formatClothing(value?: string) {
  const normalized = value?.trim()
  if (!normalized) return '未记录'
  return clothingLabels[normalized.toLowerCase()] || normalized
}
