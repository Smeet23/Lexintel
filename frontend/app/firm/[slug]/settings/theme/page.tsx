"use client"

import React, { useState, useCallback, useRef, useEffect } from "react"
import { useParams } from "next/navigation"
import { Sun, Moon } from "lucide-react"
import AppLayout from "@/layouts/AppLayout"
import PageHeader from "@/components/PageHeader"
import { Button } from "@/components/ui/button"
import AdminGuard from "@/components/theme-editor/AdminGuard"
import ThemeControlPanel from "@/components/theme-editor/ThemeControlPanel"
import ThemePreview from "@/components/theme-editor/ThemePreview"
import { useFirmThemeContext, applyThemeToDOM, loadGoogleFonts } from "@/lib/firm-theme-context"
import { useUpdateFirmTheme, useResetFirmTheme } from "@/hooks/use-firm"
import { PRESET_MAP } from "@/lib/theme-presets"
import type { ThemeConfig } from "@/lib/types"

function deepClone<T>(obj: T): T {
  return JSON.parse(JSON.stringify(obj))
}

export default function ThemeEditorPage() {
  const { slug } = useParams<{ slug: string }>()
  const { theme: savedTheme, isDarkMode, toggleDarkMode } = useFirmThemeContext()
  const updateMutation = useUpdateFirmTheme(slug)
  const resetMutation = useResetFirmTheme(slug)

  const [localTheme, setLocalTheme] = useState<ThemeConfig | null>(null)
  const savedSnapshot = useRef<string>("")

  // Initialize local theme from context
  useEffect(() => {
    if (savedTheme && !localTheme) {
      setLocalTheme(deepClone(savedTheme))
      savedSnapshot.current = JSON.stringify(savedTheme)
    }
  }, [savedTheme, localTheme])

  const hasUnsavedChanges = localTheme
    ? JSON.stringify(localTheme) !== savedSnapshot.current
    : false

  // Apply local changes to DOM for live preview
  useEffect(() => {
    if (localTheme) {
      applyThemeToDOM(localTheme, isDarkMode)
    }
  }, [localTheme, isDarkMode])

  const handleColorChange = useCallback(
    (mode: "light" | "dark", key: string, value: string) => {
      setLocalTheme((prev) => {
        if (!prev) return prev
        const next = deepClone(prev)
        next[mode][key] = value
        return next
      })
    },
    []
  )

  const handleTypographyChange = useCallback(
    (key: string, value: string) => {
      setLocalTheme((prev) => {
        if (!prev) return prev
        const next = deepClone(prev)
        next.typography = { ...next.typography, [key]: value }
        return next
      })
      // Load Google Font as user types
      if (value.trim()) {
        loadGoogleFonts([value])
      }
    },
    []
  )

  const handleLayoutChange = useCallback(
    (key: string, value: string) => {
      setLocalTheme((prev) => {
        if (!prev) return prev
        const next = deepClone(prev)
        next.layout = { ...next.layout, [key]: value }
        return next
      })
    },
    []
  )

  const handlePresetSelect = useCallback(
    (presetSlug: string) => {
      const preset =
        presetSlug === "default"
          ? PRESET_MAP["lexintel-default"]
          : PRESET_MAP[presetSlug]
      if (preset) {
        const cloned = deepClone(preset.config)
        setLocalTheme(cloned)
        applyThemeToDOM(cloned, isDarkMode)
        if (cloned.typography) {
          loadGoogleFonts(
            [
              cloned.typography["font-sans"],
              cloned.typography["font-display"],
              cloned.typography["font-mono"],
            ].filter(Boolean) as string[]
          )
        }
      }
    },
    [isDarkMode]
  )

  const handleSave = useCallback(() => {
    if (!localTheme) return
    updateMutation.mutate(localTheme, {
      onSuccess: () => {
        savedSnapshot.current = JSON.stringify(localTheme)
      },
    })
  }, [localTheme, updateMutation])

  const handleReset = useCallback(() => {
    resetMutation.mutate(undefined, {
      onSuccess: (data) => {
        const resetTheme = deepClone(data.theme)
        setLocalTheme(resetTheme)
        savedSnapshot.current = JSON.stringify(resetTheme)
        applyThemeToDOM(resetTheme, isDarkMode)
      },
    })
  }, [resetMutation, isDarkMode])

  if (!localTheme) {
    return (
      <AppLayout title="Theme Editor">
        <div className="flex items-center justify-center h-64">
          <div className="h-6 w-6 rounded-full border-2 border-foreground border-t-transparent animate-spin" />
        </div>
      </AppLayout>
    )
  }

  return (
    <AdminGuard firmSlug={slug}>
      <AppLayout title="Theme Editor">
        <PageHeader
          title="Theme Editor"
          description="Customize your firm's colors, typography, and branding"
          actions={
            <Button variant="outline" size="sm" onClick={toggleDarkMode}>
              {isDarkMode ? (
                <>
                  <Sun className="h-4 w-4" />
                  Light Mode
                </>
              ) : (
                <>
                  <Moon className="h-4 w-4" />
                  Dark Mode
                </>
              )}
            </Button>
          }
        />

        <div className="grid grid-cols-1 lg:grid-cols-5 gap-0 border border-border rounded-xl overflow-hidden bg-card shadow-elevated -mx-1">
          {/* Control Panel - left 40% */}
          <div className="lg:col-span-2 border-b lg:border-b-0 lg:border-r border-border">
            <ThemeControlPanel
              theme={localTheme}
              isDark={isDarkMode}
              onColorChange={handleColorChange}
              onTypographyChange={handleTypographyChange}
              onLayoutChange={handleLayoutChange}
              onSave={handleSave}
              onReset={handleReset}
              onPresetSelect={handlePresetSelect}
              isSaving={updateMutation.isPending}
              hasUnsavedChanges={hasUnsavedChanges}
            />
          </div>

          {/* Preview - right 60% */}
          <div className="lg:col-span-3 bg-background min-h-[600px]">
            <ThemePreview />
          </div>
        </div>
      </AppLayout>
    </AdminGuard>
  )
}
