'use client'

import { useState, useEffect, useCallback, useRef } from 'react'

export interface ProgressEvent {
  stage: string
  step: number
  total_steps: number
  progress: number
  message: string
  detail: string
  retry_attempt: number | null
  error: string | null
}

interface UseCaseProgressOptions {
  onComplete?: (event: ProgressEvent) => void
  onError?: (event: ProgressEvent) => void
}

export function useCaseProgress(
  caseId: string | null,
  options: UseCaseProgressOptions = {}
) {
  const [progress, setProgress] = useState<ProgressEvent | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [isComplete, setIsComplete] = useState(false)
  const [connectionError, setConnectionError] = useState<string | null>(null)

  const eventSourceRef = useRef<EventSource | null>(null)
  const { onComplete, onError } = options

  const disconnect = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close()
      eventSourceRef.current = null
      setIsConnected(false)
    }
  }, [])

  useEffect(() => {
    if (!caseId) {
      disconnect()
      return
    }

    const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
    const token = typeof window !== 'undefined' ? localStorage.getItem('access_token') : null

    if (!token) {
      setConnectionError('No authentication token')
      return
    }

    // Close existing connection
    disconnect()

    // Create new EventSource connection
    const url = `${API_URL}/cases/${caseId}/progress?token=${encodeURIComponent(token)}`
    const eventSource = new EventSource(url)
    eventSourceRef.current = eventSource

    eventSource.onopen = () => {
      setIsConnected(true)
      setConnectionError(null)
    }

    eventSource.addEventListener('connected', (event) => {
      setIsConnected(true)
      setConnectionError(null)
    })

    eventSource.addEventListener('progress', (event) => {
      try {
        const data: ProgressEvent = JSON.parse(event.data)
        setProgress(data)

        // Check for completion
        if (data.stage === 'ready') {
          setIsComplete(true)
          onComplete?.(data)
          eventSource.close()
        } else if (data.stage === 'error') {
          setIsComplete(true)
          onError?.(data)
          eventSource.close()
        }
      } catch (e) {
        console.error('Failed to parse progress event:', e)
      }
    })

    eventSource.onerror = (error) => {
      console.error('SSE connection error:', error)
      setIsConnected(false)

      // Don't set error if we completed successfully
      if (!isComplete) {
        setConnectionError('Connection lost. Retrying...')
      }
    }

    // Cleanup on unmount or caseId change
    return () => {
      eventSource.close()
    }
  }, [caseId, disconnect, isComplete, onComplete, onError])

  // Reset state when caseId changes
  useEffect(() => {
    if (caseId) {
      setProgress(null)
      setIsComplete(false)
      setConnectionError(null)
    }
  }, [caseId])

  return {
    progress,
    isConnected,
    isComplete,
    connectionError,
    disconnect,
  }
}

export default useCaseProgress
