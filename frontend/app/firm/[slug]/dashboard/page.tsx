"use client"

import { useParams } from "next/navigation"
import AppLayout from "@/layouts/AppLayout"
import DashboardView from "@/components/views/DashboardView"

export default function FirmDashboardPage() {
  const { slug } = useParams<{ slug: string }>()
  return (
    <AppLayout title="Dashboard">
      <DashboardView firmSlug={slug} />
    </AppLayout>
  )
}
