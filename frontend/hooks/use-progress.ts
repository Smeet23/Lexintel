import { useState, useEffect, useRef } from "react"
import { subscribeToCaseProgress } from "@/lib/api"
import type { ProgressEvent } from "@/lib/types"

export function useCaseProgress(caseId: string | undefined) {
  const [progress, setProgress] = useState<ProgressEvent | null>(null)
  const [connected, setConnected] = useState(false)
  const eventSourceRef = useRef<EventSource | null>(null)

  useEffect(() => {
    if (!caseId) return

    const es = subscribeToCaseProgress(caseId)
    eventSourceRef.current = es

    es.addEventListener("connected", () => {
      setConnected(true)
    })

    es.addEventListener("progress", (event) => {
      try {
        const data: ProgressEvent = JSON.parse(event.data)
        setProgress(data)

        if (data.stage === "ready" || data.stage === "error") {
          es.close()
          setConnected(false)
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
  }, [caseId])

  return { progress, connected }
}
