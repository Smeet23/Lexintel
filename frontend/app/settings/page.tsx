"use client"

import React, { useState } from "react"
import {
  Settings,
  Globe,
  Brain,
  Shield,
  Key,
  Database,
  Bell,
  Palette,
  Save,
} from "lucide-react"
import AppLayout from "@/layouts/AppLayout"
import PageHeader from "@/components/PageHeader"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Switch } from "@/components/ui/switch"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Separator } from "@/components/ui/separator"

interface SettingSection {
  icon: React.ReactNode
  title: string
  description: string
  children: React.ReactNode
}

function SettingCard({ icon, title, description, children }: SettingSection) {
  return (
    <div className="bg-white rounded-xl border border-border shadow-elevated p-5 sm:p-8">
      <div className="flex items-start gap-3 sm:gap-4 mb-6">
        <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-surface shrink-0">
          {icon}
        </div>
        <div>
          <h3 className="font-display font-semibold text-foreground">{title}</h3>
          <p className="text-sm text-muted mt-0.5">{description}</p>
        </div>
      </div>
      <div className="space-y-5">{children}</div>
    </div>
  )
}

function SettingRow({
  label,
  description,
  children,
}: {
  label: string
  description?: string
  children: React.ReactNode
}) {
  return (
    <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 sm:gap-4 py-3">
      <div>
        <p className="text-sm font-medium text-foreground">{label}</p>
        {description && <p className="text-xs text-muted mt-0.5">{description}</p>}
      </div>
      <div className="shrink-0">{children}</div>
    </div>
  )
}

export default function SettingsPage() {
  const [jurisdiction, setJurisdiction] = useState("us-federal")
  const [model, setModel] = useState("standard")
  const [retention, setRetention] = useState("1-year")
  const [advancedModel, setAdvancedModel] = useState(false)
  const [emailNotifs, setEmailNotifs] = useState(true)
  const [queryNotifs, setQueryNotifs] = useState(false)
  const [darkMode, setDarkMode] = useState(false)

  return (
    <AppLayout title="Settings">
      <PageHeader
        title="Settings"
        description="Configure your workspace preferences and security settings"
        actions={
          <Button>
            <Save className="h-4 w-4" />
            Save Changes
          </Button>
        }
      />

      <div className="max-w-3xl space-y-8">
        {/* Jurisdiction */}
        <SettingCard
          icon={<Globe className="h-4 w-4 text-foreground" />}
          title="Jurisdiction Default"
          description="Set the default jurisdiction for new matters and searches"
        >
          <Select value={jurisdiction} onValueChange={setJurisdiction}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="us-federal">US - Federal</SelectItem>
              <SelectItem value="us-california">US - California</SelectItem>
              <SelectItem value="us-new-york">US - New York</SelectItem>
              <SelectItem value="us-delaware">US - Delaware</SelectItem>
              <SelectItem value="uk">United Kingdom</SelectItem>
              <SelectItem value="eu">European Union</SelectItem>
              <SelectItem value="eu-gdpr">EU - GDPR Specific</SelectItem>
            </SelectContent>
          </Select>
        </SettingCard>

        {/* AI Model */}
        <SettingCard
          icon={<Brain className="h-4 w-4 text-foreground" />}
          title="AI Model"
          description="Choose the AI model for document analysis and queries"
        >
          <Select value={model} onValueChange={setModel}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="standard">Standard (GPT-4o) - Balanced speed and accuracy</SelectItem>
              <SelectItem value="advanced">Advanced (GPT-4o + Reranking) - Higher accuracy</SelectItem>
              <SelectItem value="fast">Fast (GPT-4o Mini) - Faster responses, lower cost</SelectItem>
            </SelectContent>
          </Select>

          <Separator />

          <SettingRow
            label="Enable Advanced Model"
            description="Uses additional reranking for more accurate citations"
          >
            <Switch checked={advancedModel} onCheckedChange={setAdvancedModel} />
          </SettingRow>

          <SettingRow
            label="Confidence Threshold"
            description="Minimum confidence score to display in results"
          >
            <Input type="number" defaultValue="60" className="w-20 h-8 text-sm text-center" min={0} max={100} />
          </SettingRow>
        </SettingCard>

        {/* Data Retention */}
        <SettingCard
          icon={<Database className="h-4 w-4 text-foreground" />}
          title="Data Retention"
          description="Control how long your data is stored"
        >
          <Select value={retention} onValueChange={setRetention}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="30-days">30 Days</SelectItem>
              <SelectItem value="90-days">90 Days</SelectItem>
              <SelectItem value="1-year">1 Year</SelectItem>
              <SelectItem value="indefinite">Indefinite</SelectItem>
            </SelectContent>
          </Select>

          <div className="rounded-lg bg-amber-50 border border-amber-200/60 p-3">
            <p className="text-[12px] text-amber-700">
              <strong>Note:</strong> Changing retention settings will not retroactively delete existing data. Contact support for data deletion requests.
            </p>
          </div>
        </SettingCard>

        {/* Security */}
        <SettingCard
          icon={<Shield className="h-4 w-4 text-foreground" />}
          title="Security"
          description="Manage authentication and security preferences"
        >
          <SettingRow
            label="Two-Factor Authentication"
            description="Require 2FA for account access"
          >
            <Switch defaultChecked />
          </SettingRow>

          <Separator />

          <SettingRow
            label="Session Timeout"
            description="Auto-logout after inactivity"
          >
            <Select defaultValue="4h">
              <SelectTrigger className="w-32">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="30m">30 Minutes</SelectItem>
                <SelectItem value="1h">1 Hour</SelectItem>
                <SelectItem value="4h">4 Hours</SelectItem>
                <SelectItem value="8h">8 Hours</SelectItem>
              </SelectContent>
            </Select>
          </SettingRow>
        </SettingCard>

        {/* API Keys */}
        <SettingCard
          icon={<Key className="h-4 w-4 text-foreground" />}
          title="API Integration"
          description="Manage API keys for enterprise integrations"
        >
          <div className="flex flex-col sm:flex-row gap-2">
            <Input
              value="vrt_sk_••••••••••••••••••••••••"
              disabled
              className="font-mono text-sm"
            />
            <Button variant="outline" size="sm" className="shrink-0">Regenerate</Button>
          </div>
          <p className="text-xs text-muted">
            Last regenerated: January 15, 2026
          </p>
        </SettingCard>

        {/* Notifications */}
        <SettingCard
          icon={<Bell className="h-4 w-4 text-foreground" />}
          title="Notifications"
          description="Configure email and in-app notification preferences"
        >
          <SettingRow
            label="Email Notifications"
            description="Receive weekly usage summaries via email"
          >
            <Switch checked={emailNotifs} onCheckedChange={setEmailNotifs} />
          </SettingRow>

          <Separator />

          <SettingRow
            label="Query Completion Alerts"
            description="Notify when long-running queries complete"
          >
            <Switch checked={queryNotifs} onCheckedChange={setQueryNotifs} />
          </SettingRow>
        </SettingCard>

        {/* Appearance */}
        <SettingCard
          icon={<Palette className="h-4 w-4 text-foreground" />}
          title="Appearance"
          description="Customize the look and feel of your workspace"
        >
          <SettingRow
            label="Dark Mode"
            description="Switch to dark theme (coming soon)"
          >
            <Switch checked={darkMode} onCheckedChange={setDarkMode} disabled />
          </SettingRow>
        </SettingCard>
      </div>
    </AppLayout>
  )
}
