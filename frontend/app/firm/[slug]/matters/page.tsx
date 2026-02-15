"use client"

import { useParams } from "next/navigation"
import AppLayout from "@/layouts/AppLayout"
import MattersView from "@/components/views/MattersView"

export default function FirmMattersPage() {
  const { slug } = useParams<{ slug: string }>()
  return (
    <AppLayout title="Matters">
      <MattersView firmSlug={slug} />
    </AppLayout>
  )
}
