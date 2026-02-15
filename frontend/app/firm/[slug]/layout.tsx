"use client"

import FirmThemeProvider from "@/lib/firm-theme-context"
import { useParams } from "next/navigation"

export default function FirmLayout({ children }: { children: React.ReactNode }) {
  const { slug } = useParams<{ slug: string }>()

  return (
    <FirmThemeProvider firmSlug={slug}>
      {children}
    </FirmThemeProvider>
  )
}
