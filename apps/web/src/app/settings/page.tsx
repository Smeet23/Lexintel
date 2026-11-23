"use client"

import AppLayout from "@/layouts/AppLayout"
import PageHeader from "@/components/PageHeader"
import ComingSoon from "@/components/ComingSoon"
import { Settings } from "lucide-react"

export default function SettingsPage() {
  return (
    <AppLayout title="Settings">
      <PageHeader title="Settings" description="Configure your workspace preferences" />
      <ComingSoon icon={Settings} description="Workspace preferences such as default jurisdiction and data retention." />
    </AppLayout>
  )
}
