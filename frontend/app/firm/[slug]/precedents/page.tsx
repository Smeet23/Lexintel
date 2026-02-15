"use client"

import { useParams } from "next/navigation"
import AppLayout from "@/layouts/AppLayout"
import PrecedentsView from "@/components/views/PrecedentsView"

export default function FirmPrecedentsPage() {
  const { slug } = useParams<{ slug: string }>()
  return (
    <AppLayout title="Precedents">
      <PrecedentsView firmSlug={slug} />
    </AppLayout>
  )
}
