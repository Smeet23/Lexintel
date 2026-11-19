"use client"

import { AlertCircle } from "lucide-react"
import { Button } from "@/components/ui/button"

interface ErrorStateProps {
  onRetry: () => void
  className?: string
}

export default function ErrorState({ onRetry, className = "" }: ErrorStateProps) {
  return (
    <div className={`flex flex-col items-center justify-center py-12 px-6 text-center ${className}`}>
      <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-surface mb-3">
        <AlertCircle className="h-5 w-5 text-muted" />
      </div>
      <p className="text-sm font-medium text-foreground">Couldn&apos;t reach the server</p>
      <p className="text-xs text-muted mt-1">Check that the API is running and try again.</p>
      <Button variant="outline" size="sm" className="mt-4" onClick={onRetry}>Retry</Button>
    </div>
  )
}
