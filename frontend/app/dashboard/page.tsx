"use client"

import AppLayout from "@/layouts/AppLayout"
import DashboardView from "@/components/views/DashboardView"

export default function DashboardPage() {
  return (
    <AppLayout title="Dashboard">
      <DashboardView />
    </AppLayout>
  )
}
