"use client"

import AppLayout from "@/layouts/AppLayout"
import PageHeader from "@/components/PageHeader"

export default function FirmTeamPage() {
  return (
    <AppLayout title="Team">
      <PageHeader
        title="Team Management"
        description="Manage your firm's team members"
      />
      <p className="text-muted-foreground">Team management for firm-scoped routes coming soon.</p>
    </AppLayout>
  )
}
