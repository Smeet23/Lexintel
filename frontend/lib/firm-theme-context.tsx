"use client"

import React, { createContext, useContext, useEffect, useState, useCallback } from "react"
import { useQuery } from "@tanstack/react-query"
import { getFirmTheme } from "@/lib/api-services"
import { PRESET_MAP } from "@/lib/theme-presets"
import type { ThemeConfig, ThemeColorTokens, FirmThemeResponse } from "@/lib/types"

const FALLBACK_THEME: ThemeConfig = PRESET_MAP["lexintel-default"].config

interface FirmThemeContextValue {
  firmName: string
  firmSlug: string
  logoUrl: string | null
  theme: ThemeConfig | null
  isDarkMode: boolean
  toggleDarkMode: () => void
  isLoading: boolean
}

const FirmThemeContext = createContext<FirmThemeContextValue | null>(null)

export function useFirmThemeContext() {
  const ctx = useContext(FirmThemeContext)
  if (!ctx) throw new Error("useFirmThemeContext must be used within FirmThemeProvider")
  return ctx
}

/**
 * Try to get firm theme context, returns null if not within a FirmThemeProvider.
 */
export function useFirmThemeContextSafe(): FirmThemeContextValue | null {
  return useContext(FirmThemeContext)
}

/**
 * Apply theme to DOM by setting --app-* CSS variables.
 *
 * We set --app-* (not --color-*) because @theme inline maps
 * --color-primary: var(--app-primary). Inline styles on
 * documentElement have higher specificity than @layer theme rules.
 */
export function applyThemeToDOM(theme: ThemeConfig, isDark: boolean) {
  const root = document.documentElement
  const colors: ThemeColorTokens = isDark ? theme.dark : theme.light

  for (const [key, value] of Object.entries(colors)) {
    root.style.setProperty(`--app-${key}`, value)
  }

  if (theme.typography) {
    const sans = theme.typography["font-sans"]
    const display = theme.typography["font-display"]
    const mono = theme.typography["font-mono"]
    if (sans) root.style.setProperty("--app-font-sans", `"${sans}", ui-sans-serif, system-ui, -apple-system, sans-serif`)
    if (display) root.style.setProperty("--app-font-display", `"${display}", Georgia, "Times New Roman", serif`)
    if (mono) root.style.setProperty("--app-font-mono", `"${mono}", ui-monospace, monospace`)
  }

  if (theme.layout) {
    for (const [key, value] of Object.entries(theme.layout)) {
      root.style.setProperty(`--app-${key}`, value)
    }
  }

  if (isDark) {
    root.classList.add("dark")
  } else {
    root.classList.remove("dark")
  }
}

export function clearThemeFromDOM() {
  const root = document.documentElement
  const style = root.style
  for (let i = style.length - 1; i >= 0; i--) {
    const prop = style[i]
    if (prop.startsWith("--app-")) {
      style.removeProperty(prop)
    }
  }
  root.classList.remove("dark")
}

export function loadGoogleFonts(fonts: string[]) {
  const filtered = fonts.filter(Boolean)
  if (filtered.length === 0) return

  const linkId = "firm-theme-fonts"
  let link = document.getElementById(linkId) as HTMLLinkElement | null
  if (!link) {
    link = document.createElement("link")
    link.id = linkId
    link.rel = "stylesheet"
    document.head.appendChild(link)
  }

  const families = filtered
    .map((f) => `family=${f.replace(/ /g, "+")}:wght@400;500;600;700`)
    .join("&")
  link.href = `https://fonts.googleapis.com/css2?${families}&display=swap`
}

interface FirmThemeProviderProps {
  children: React.ReactNode
  firmSlug: string
}

export default function FirmThemeProvider({ children, firmSlug }: FirmThemeProviderProps) {
  const [isDarkMode, setIsDarkMode] = useState(false)

  const { data, isLoading, isError } = useQuery<FirmThemeResponse>({
    queryKey: ["firm-theme", firmSlug],
    queryFn: () => getFirmTheme(firmSlug),
    staleTime: 5 * 60 * 1000,
    retry: 1,
  })

  const activeTheme = data?.theme ?? (isError ? FALLBACK_THEME : null)

  useEffect(() => {
    const stored = localStorage.getItem(`lexintel_dark_mode_${firmSlug}`)
    if (stored === "true") setIsDarkMode(true)
  }, [firmSlug])

  useEffect(() => {
    if (activeTheme) {
      applyThemeToDOM(activeTheme, isDarkMode)
      loadGoogleFonts([
        activeTheme.typography?.["font-sans"],
        activeTheme.typography?.["font-display"],
        activeTheme.typography?.["font-mono"],
      ].filter(Boolean) as string[])
    }
    return () => clearThemeFromDOM()
  }, [activeTheme, isDarkMode])

  const toggleDarkMode = useCallback(() => {
    setIsDarkMode((prev) => {
      const next = !prev
      localStorage.setItem(`lexintel_dark_mode_${firmSlug}`, String(next))
      return next
    })
  }, [firmSlug])

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center" style={{ backgroundColor: "#FAFAF8" }}>
        <div className="flex flex-col items-center gap-3">
          <div className="h-8 w-8 rounded-lg animate-pulse" style={{ backgroundColor: "#111" }} />
          <div className="h-4 w-32 rounded animate-pulse" style={{ backgroundColor: "#E5E4E2" }} />
        </div>
      </div>
    )
  }

  return (
    <FirmThemeContext.Provider
      value={{
        firmName: data?.firm_name || "",
        firmSlug,
        logoUrl: data?.logo_url || null,
        theme: activeTheme,
        isDarkMode,
        toggleDarkMode,
        isLoading,
      }}
    >
      {children}
    </FirmThemeContext.Provider>
  )
}
