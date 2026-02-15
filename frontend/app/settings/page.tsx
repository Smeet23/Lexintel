"use client"

import AppLayout from "@/layouts/AppLayout"
import SettingsView from "@/components/views/SettingsView"

export default function SettingsPage() {
  return (
    <AppLayout title="Settings">
      <SettingsView />
    </AppLayout>
  )
}
