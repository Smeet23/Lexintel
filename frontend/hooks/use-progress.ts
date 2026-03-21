import { useState, useEffect, useRef } from "react"
import { useQueryClient } from "@tanstack/react-query"
import { subscribeMatterProgress } from "@/lib/api-services"
import type { ProgressEvent } from "@/lib/types"

export function useMatterProgress(matterId: string | undefined, isProcessing: boolean) {
  const [progress, setProgress] = useState<ProgressEvent | null>(null)
  const [connected, setConnected] = useState(false)
  const eventSourceRef = useRef<EventSource | null>(null)
  const queryClient = useQueryClient()

  useEffect(() => {
    // Only connect when the matter is actually processing
    if (!matterId || !isProcessing) {
      setProgress(null)
      setConnected(false)
      return
    }

    const es = subscribeMatterProgress(matterId)
    eventSourceRef.current = es

    es.addEventListener("connected", () => {
      setConnected(true)
    })

    es.addEventListener("progress", (event) => {
      try {
        const data: ProgressEvent = JSON.parse(event.data)
        // Only update if progress moves forward (prevent backwards jumps)
        setProgress((prev) => {
          if (!prev) return data
          if (data.overall_progress >= prev.overall_progress) return data
          return prev
        })

        if (data.stage === "ready" || data.stage === "error") {
          es.close()
          setConnected(false)
          // Refresh matter data to pick up new status
          queryClient.invalidateQueries({ queryKey: ["matters", matterId] })
          queryClient.invalidateQueries({ queryKey: ["matters", matterId, "documents"] })
          queryClient.invalidateQueries({ queryKey: ["matters"] })
        }
      } catch {
        // ignore malformed events
      }
    })

    es.onerror = () => {
      es.close()
      setConnected(false)
    }

    return () => {
      es.close()
      setConnected(false)
    }
  }, [matterId, isProcessing, queryClient])

  return { progress, connected }
}
