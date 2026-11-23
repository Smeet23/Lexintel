"use client"

import AppLayout from "@/layouts/AppLayout"
import PageHeader from "@/components/PageHeader"
import ComingSoon from "@/components/ComingSoon"
import { CreditCard } from "lucide-react"

export default function BillingPage() {
  return (
    <AppLayout title="Billing">
      <PageHeader title="Billing & Usage" description="Plans, usage and invoices" />
      <ComingSoon icon={CreditCard} description="Plans, usage and invoices." />
    </AppLayout>
  )
}
