/**
 * WCAG 2.0 contrast ratio utilities for theme validation.
 *
 * AA requires 4.5:1 for normal text, 3:1 for large text.
 * AAA requires 7:1 for normal text, 4.5:1 for large text.
 */

function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace("#", "")
  return [
    parseInt(h.substring(0, 2), 16),
    parseInt(h.substring(2, 4), 16),
    parseInt(h.substring(4, 6), 16),
  ]
}

function srgbToLinear(c: number): number {
  const s = c / 255
  return s <= 0.04045 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4)
}

function relativeLuminance(hex: string): number {
  const [r, g, b] = hexToRgb(hex)
  return 0.2126 * srgbToLinear(r) + 0.7152 * srgbToLinear(g) + 0.0722 * srgbToLinear(b)
}

export function contrastRatio(hex1: string, hex2: string): number {
  const l1 = relativeLuminance(hex1)
  const l2 = relativeLuminance(hex2)
  const lighter = Math.max(l1, l2)
  const darker = Math.min(l1, l2)
  return (lighter + 0.05) / (darker + 0.05)
}

export type ContrastLevel = "AAA" | "AA" | "fail"

export function checkContrast(fg: string, bg: string): { ratio: number; level: ContrastLevel } {
  const ratio = contrastRatio(fg, bg)
  const level: ContrastLevel = ratio >= 7 ? "AAA" : ratio >= 4.5 ? "AA" : "fail"
  return { ratio, level }
}

/** Key token pairs that must meet AA contrast (4.5:1). */
const CRITICAL_PAIRS: [string, string, string][] = [
  ["foreground", "background", "Text on background"],
  ["primary-foreground", "primary", "Text on primary"],
  ["accent-foreground", "accent", "Text on accent"],
  ["card-foreground", "card", "Text on card"],
  ["destructive-foreground", "destructive", "Text on destructive"],
  ["muted-foreground", "background", "Muted text on background"],
  ["sidebar-foreground", "sidebar-background", "Sidebar text"],
]

export interface ContrastWarning {
  pair: string
  fg: string
  bg: string
  ratio: number
  level: ContrastLevel
}

export function validateThemeContrast(
  colors: Record<string, string>
): ContrastWarning[] {
  const warnings: ContrastWarning[] = []

  for (const [fgKey, bgKey, label] of CRITICAL_PAIRS) {
    const fg = colors[fgKey]
    const bg = colors[bgKey]
    if (!fg || !bg) continue
    if (!/^#[0-9a-fA-F]{6}$/.test(fg) || !/^#[0-9a-fA-F]{6}$/.test(bg)) continue

    const { ratio, level } = checkContrast(fg, bg)
    if (level === "fail") {
      warnings.push({ pair: label, fg, bg, ratio, level })
    }
  }

  return warnings
}
