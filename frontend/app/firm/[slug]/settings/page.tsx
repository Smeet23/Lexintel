"use client"

import { useParams } from "next/navigation"
import AppLayout from "@/layouts/AppLayout"
import SettingsView from "@/components/views/SettingsView"

export default function FirmSettingsPage() {
  const { slug } = useParams<{ slug: string }>()
  return (
    <AppLayout title="Settings">
      <SettingsView firmSlug={slug} />
    </AppLayout>
  )
}
