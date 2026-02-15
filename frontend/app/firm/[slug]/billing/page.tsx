"use client"

import AppLayout from "@/layouts/AppLayout"
import PageHeader from "@/components/PageHeader"

export default function FirmBillingPage() {
  return (
    <AppLayout title="Billing">
      <PageHeader
        title="Billing & Usage"
        description="Monitor token usage and manage your plan"
      />
      <p className="text-muted-foreground">Billing for firm-scoped routes coming soon.</p>
    </AppLayout>
  )
}
