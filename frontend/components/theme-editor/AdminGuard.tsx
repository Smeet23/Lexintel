"use client"

import React from "react"
import { Shield } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useRouter } from "next/navigation"

interface AdminGuardProps {
  children: React.ReactNode
  firmSlug: string
}

export default function AdminGuard({ children, firmSlug }: AdminGuardProps) {
  // TODO: Check user role from API/context
  // For now, always allow access (demo mode)
  const isAdmin = true
  const router = useRouter()

  if (!isAdmin) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="text-center max-w-sm">
          <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-destructive/10 mx-auto mb-4">
            <Shield className="h-6 w-6 text-destructive" />
          </div>
          <h2 className="text-lg font-display font-semibold text-foreground">Access Denied</h2>
          <p className="text-sm text-muted-foreground mt-2">
            You need admin privileges to access the theme editor.
          </p>
          <Button
            variant="outline"
            className="mt-4"
            onClick={() => router.push(`/firm/${firmSlug}/settings`)}
          >
            Back to Settings
          </Button>
        </div>
      </div>
    )
  }

  return <>{children}</>
}
