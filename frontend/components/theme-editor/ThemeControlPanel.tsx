"use client"

import React, { useState } from "react"
import { Palette, Type, Layout, RotateCcw, Save, Loader2, AlertTriangle } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import ColorPicker from "./ColorPicker"
import { validateThemeContrast } from "@/lib/contrast"
import type { ThemeConfig } from "@/lib/types"

const COLOR_GROUPS = [
  {
    name: "Brand",
    tokens: ["primary", "primary-foreground", "primary-light", "accent", "accent-foreground", "accent-hover", "accent-muted"],
  },
  {
    name: "Base",
    tokens: ["background", "foreground", "card", "card-foreground", "popover", "popover-foreground"],
  },
  {
    name: "Surface",
    tokens: ["surface", "surface-hover", "secondary", "secondary-foreground"],
  },
  {
    name: "Subtle",
    tokens: ["muted", "muted-foreground", "border", "border-strong", "input", "ring"],
  },
  {
    name: "Semantic",
    tokens: ["destructive", "destructive-foreground", "success", "success-light", "warning", "warning-light"],
  },
  {
    name: "Sidebar",
    tokens: ["sidebar-background", "sidebar-foreground", "sidebar-accent", "sidebar-border"],
  },
  {
    name: "Charts",
    tokens: ["chart-1", "chart-2", "chart-3", "chart-4", "chart-5"],
  },
]

interface ThemeControlPanelProps {
  theme: ThemeConfig
  isDark: boolean
  onColorChange: (mode: "light" | "dark", key: string, value: string) => void
  onTypographyChange: (key: string, value: string) => void
  onLayoutChange: (key: string, value: string) => void
  onSave: () => void
  onReset: () => void
  onPresetSelect: (presetSlug: string) => void
  isSaving: boolean
  hasUnsavedChanges: boolean
}

export default function ThemeControlPanel({
  theme,
  isDark,
  onColorChange,
  onTypographyChange,
  onLayoutChange,
  onSave,
  onReset,
  onPresetSelect,
  isSaving,
  hasUnsavedChanges,
}: ThemeControlPanelProps) {
  const [activeTab, setActiveTab] = useState<"colors" | "typography" | "layout">("colors")
  const colors = isDark ? theme.dark : theme.light
  const mode = isDark ? "dark" : "light"
  const contrastWarnings = validateThemeContrast(colors)

  const tabs = [
    { id: "colors" as const, label: "Colors", icon: Palette },
    { id: "typography" as const, label: "Typography", icon: Type },
    { id: "layout" as const, label: "Layout", icon: Layout },
  ]

  return (
    <div className="flex flex-col h-full">
      {/* Action Bar */}
      <div className="flex items-center justify-between p-4 border-b border-border">
        <Select onValueChange={onPresetSelect}>
          <SelectTrigger className="w-44 h-8 text-xs">
            <SelectValue placeholder="Apply preset..." />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="default">Lexintel Default</SelectItem>
            <SelectItem value="corporate-navy">Corporate Navy</SelectItem>
            <SelectItem value="modern-slate">Modern Slate</SelectItem>
            <SelectItem value="warm-counsel">Warm Counsel</SelectItem>
            <SelectItem value="pacific-blue">Pacific Blue</SelectItem>
            <SelectItem value="emerald-ivory">Emerald & Ivory</SelectItem>
            <SelectItem value="minimal-charcoal">Minimal Charcoal</SelectItem>
            <SelectItem value="burgundy-classic">Burgundy Classic</SelectItem>
          </SelectContent>
        </Select>
        <div className="flex items-center gap-2">
          {hasUnsavedChanges && (
            <span className="text-xs text-amber-600 font-medium">Unsaved</span>
          )}
          <Button variant="outline" size="sm" onClick={onReset} className="h-8 text-xs">
            <RotateCcw className="h-3 w-3" />
            Reset
          </Button>
          <Button size="sm" onClick={onSave} disabled={isSaving || !hasUnsavedChanges} className="h-8 text-xs">
            {isSaving ? <Loader2 className="h-3 w-3 animate-spin" /> : <Save className="h-3 w-3" />}
            Save
          </Button>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-border">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center gap-1.5 px-4 py-2.5 text-xs font-medium transition-colors border-b-2 ${
              activeTab === tab.id
                ? "border-foreground text-foreground"
                : "border-transparent text-muted-foreground hover:text-foreground"
            }`}
          >
            <tab.icon className="h-3.5 w-3.5" />
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        {activeTab === "colors" && (
          <>
            {contrastWarnings.length > 0 && (
              <div className="rounded-lg border border-amber-300 bg-amber-50 p-3 space-y-1.5">
                <div className="flex items-center gap-1.5 text-amber-700">
                  <AlertTriangle className="h-3.5 w-3.5 shrink-0" />
                  <span className="text-xs font-semibold">Contrast warnings (WCAG AA)</span>
                </div>
                {contrastWarnings.map((w) => (
                  <p key={w.pair} className="text-[11px] text-amber-600 ml-5">
                    {w.pair}: {w.ratio.toFixed(1)}:1 (needs 4.5:1)
                  </p>
                ))}
              </div>
            )}
            {COLOR_GROUPS.map((group) => (
              <div key={group.name}>
                <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">
                  {group.name}
                </h4>
                <div className="grid grid-cols-2 gap-3">
                  {group.tokens.map((token) => (
                    <ColorPicker
                      key={token}
                      tokenKey={token}
                      value={colors[token] || "#000000"}
                      label={token}
                      onChange={(key, value) => onColorChange(mode, key, value)}
                    />
                  ))}
                </div>
              </div>
            ))}
          </>
        )}

        {activeTab === "typography" && (
          <div className="space-y-4">
            <div>
              <label className="text-xs font-medium text-foreground mb-1.5 block">Sans Serif Font</label>
              <Input
                value={theme.typography?.["font-sans"] || ""}
                onChange={(e) => onTypographyChange("font-sans", e.target.value)}
                placeholder="DM Sans"
                className="text-sm"
              />
            </div>
            <div>
              <label className="text-xs font-medium text-foreground mb-1.5 block">Display Font</label>
              <Input
                value={theme.typography?.["font-display"] || ""}
                onChange={(e) => onTypographyChange("font-display", e.target.value)}
                placeholder="DM Serif Display"
                className="text-sm"
              />
            </div>
            <div>
              <label className="text-xs font-medium text-foreground mb-1.5 block">Monospace Font</label>
              <Input
                value={theme.typography?.["font-mono"] || ""}
                onChange={(e) => onTypographyChange("font-mono", e.target.value)}
                placeholder="JetBrains Mono"
                className="text-sm"
              />
            </div>
            <div className="rounded-lg border border-border p-4 mt-4">
              <p className="text-xs text-muted-foreground mb-2">Preview</p>
              <p className="text-lg font-display">Display heading</p>
              <p className="text-sm mt-1">Body text in sans-serif</p>
              <p className="text-xs font-mono mt-1 text-muted-foreground">Monospace code</p>
            </div>
          </div>
        )}

        {activeTab === "layout" && (
          <div className="space-y-4">
            <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Border Radius</h4>
            {["radius-sm", "radius-md", "radius-lg", "radius-xl"].map((key) => (
              <div key={key} className="flex items-center gap-3">
                <label className="text-xs text-muted-foreground w-20">{key}</label>
                <Input
                  value={theme.layout?.[key as keyof typeof theme.layout] || ""}
                  onChange={(e) => onLayoutChange(key, e.target.value)}
                  className="text-sm flex-1"
                  placeholder="0.5rem"
                />
                <div
                  className="h-8 w-8 border-2 border-foreground"
                  style={{ borderRadius: theme.layout?.[key as keyof typeof theme.layout] || "0.5rem" }}
                />
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
