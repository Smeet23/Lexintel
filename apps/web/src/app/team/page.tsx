"use client"

import AppLayout from "@/layouts/AppLayout"
import PageHeader from "@/components/PageHeader"
import ComingSoon from "@/components/ComingSoon"
import { Users } from "lucide-react"

export default function TeamPage() {
  return (
    <AppLayout title="Team">
      <PageHeader title="Team Management" description="Manage workspace members and roles" />
      <ComingSoon icon={Users} description="Invite colleagues and manage roles." />
    </AppLayout>
  )
}
