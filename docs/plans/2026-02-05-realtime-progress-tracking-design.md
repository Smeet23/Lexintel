# Real-Time Document Processing Progress Tracking

**Date:** 2026-02-05
**Status:** Approved for Implementation

## Overview

Add real-time progress tracking for document processing with a visual stepper UI, using Redis Pub/Sub for backend messaging and SSE for browser delivery.

## Problem

Currently, users only see "processing" or "ready" status. They have no visibility into:
- Which processing stage the document is at
- How long it might take
- Whether retries are happening
- What went wrong if it fails

## Solution

### Architecture: Redis Pub/Sub + SSE

```
┌─────────────────┐     Redis Pub/Sub      ┌─────────────────┐        SSE         ┌─────────────────┐
│  Celery Worker  │ ──────────────────────►│   FastAPI API   │ ──────────────────►│     Browser     │
│                 │   channel: case:{id}   │                 │   /cases/{id}/     │                 │
│  - processes    │                        │  - subscribes   │     progress       │  - displays     │
│  - publishes    │                        │  - streams      │                    │    stepper      │
└─────────────────┘                        └─────────────────┘                    └─────────────────┘
                                                   │
                                                   │ Only final status
                                                   ▼
                                           ┌─────────────────┐
                                           │   PostgreSQL    │
                                           │  (status field) │
                                           └─────────────────┘
```

**Why this approach:**
- Redis handles fast pub/sub between Celery and FastAPI (no DB writes for progress)
- SSE handles browser delivery (simple, auto-reconnect, works through proxies)
- Database only stores final status, not transient progress

## Processing Stages

| Step | Stage Key | Display Name | Has Progress % | Description |
|------|-----------|--------------|----------------|-------------|
| 1 | `uploaded` | Uploaded | No | File received, stored in blob |
| 2 | `downloading` | Downloading | No | Fetching from blob storage |
| 3 | `chunking` | Chunking document | Yes | Splitting document into chunks |
| 4 | `embedding` | Generating embeddings | Yes | Creating vector embeddings |
| 5 | `indexing` | Indexing vectors | Yes | Upserting to Qdrant |
| 6 | `storing` | Storing metadata | No | Saving to PostgreSQL |
| 7 | `ready` | Ready | No | Complete, ready for queries |

Special states:
- `error` - Processing failed after all retries
- `retrying` - Retry in progress (shows attempt number)

## Data Structures

### Redis Channel

```
Channel: lexintel:case:{case_id}:progress
```

### Progress Event Payload

```json
{
  "stage": "embedding",
  "step": 4,
  "total_steps": 7,
  "progress": 45,
  "message": "Generating embeddings...",
  "detail": "54 of 120 chunks",
  "retry_attempt": null,
  "error": null
}
```

### Error Event Payload

```json
{
  "stage": "error",
  "step": 4,
  "total_steps": 7,
  "progress": 0,
  "message": "Processing failed",
  "detail": "Failed at embedding stage",
  "retry_attempt": 3,
  "error": "OpenAI API rate limit exceeded"
}
```

## Backend Implementation

### New Files

#### `backend/services/progress.py`

```python
import redis
import json
from backend.config import get_settings

settings = get_settings()
redis_client = redis.from_url(settings.celery_broker_url)

STAGES = [
    ("uploaded", "Uploaded"),
    ("downloading", "Downloading"),
    ("chunking", "Chunking document"),
    ("embedding", "Generating embeddings"),
    ("indexing", "Indexing vectors"),
    ("storing", "Storing metadata"),
    ("ready", "Ready"),
]

def get_step_number(stage: str) -> int:
    for i, (key, _) in enumerate(STAGES):
        if key == stage:
            return i + 1
    return 0

def publish_progress(
    case_id: str,
    stage: str,
    progress: int = 0,
    message: str = "",
    detail: str = "",
    retry_attempt: int = None,
    error: str = None
):
    """Publish progress event to Redis channel."""
    channel = f"lexintel:case:{case_id}:progress"
    payload = {
        "stage": stage,
        "step": get_step_number(stage),
        "total_steps": len(STAGES),
        "progress": progress,
        "message": message or dict(STAGES).get(stage, stage),
        "detail": detail,
        "retry_attempt": retry_attempt,
        "error": error,
    }
    redis_client.publish(channel, json.dumps(payload))
```

### Modified Files

#### `backend/tasks.py` (key changes)

```python
from backend.services.progress import publish_progress

@shared_task(bind=True, max_retries=3, ...)
def process_document_task(self, case_id: str):
    try:
        # Stage 2: Downloading
        publish_progress(case_id, "downloading", message="Fetching document from storage...")
        document_content = download_document_from_blob(case.blob_storage_path)

        # Stage 3: Chunking
        publish_progress(case_id, "chunking", progress=0, message="Splitting document into chunks...")
        chunks = chunk_document_from_blob(document_content, file_type=file_type)
        publish_progress(case_id, "chunking", progress=100, detail=f"{len(chunks)} chunks created")

        # Stage 4: Embedding (with batch progress)
        publish_progress(case_id, "embedding", progress=0, message="Generating embeddings...")
        embeddings = []
        batch_size = 20
        for i in range(0, len(chunks), batch_size):
            batch = chunk_contents[i:i+batch_size]
            batch_embeddings = embed_chunks(batch)
            embeddings.extend(batch_embeddings)
            progress = min(100, int((i + len(batch)) / len(chunks) * 100))
            publish_progress(case_id, "embedding", progress=progress,
                           detail=f"{i + len(batch)} of {len(chunks)} chunks")

        # Stage 5: Indexing
        publish_progress(case_id, "indexing", progress=0, message="Creating vector index...")
        create_collection(case_id)
        publish_progress(case_id, "indexing", progress=50, detail="Collection created")
        upsert_vectors(case_id=case_id, chunks=chunks_with_ids, embeddings=embeddings)
        publish_progress(case_id, "indexing", progress=100, detail=f"{len(chunks)} vectors indexed")

        # Stage 6: Storing
        publish_progress(case_id, "storing", message="Saving metadata...")
        # ... store chunks in DB

        # Stage 7: Ready
        case.status = "ready"
        db.commit()
        publish_progress(case_id, "ready", progress=100,
                        message="Processing complete!",
                        detail=f"{len(chunks)} chunks ready for analysis")

    except Exception as exc:
        retry_count = self.request.retries
        if retry_count < self.max_retries:
            publish_progress(case_id, "retrying",
                           retry_attempt=retry_count + 1,
                           message=f"Retrying... (attempt {retry_count + 1}/{self.max_retries})",
                           error=str(exc))
            raise self.retry(exc=exc)
        else:
            publish_progress(case_id, "error",
                           retry_attempt=retry_count,
                           message="Processing failed",
                           error=str(exc))
```

#### `backend/main.py` (add SSE endpoint)

```python
import asyncio
import redis.asyncio as aioredis
from sse_starlette.sse import EventSourceResponse

@app.get("/cases/{case_id}/progress")
async def case_progress_stream(
    case_id: str,
    current_user_id: UUID = Depends(get_current_user)
):
    """SSE endpoint for real-time progress updates."""

    async def event_generator():
        redis = await aioredis.from_url(settings.celery_broker_url)
        pubsub = redis.pubsub()
        await pubsub.subscribe(f"lexintel:case:{case_id}:progress")

        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    data = message["data"]
                    if isinstance(data, bytes):
                        data = data.decode("utf-8")
                    yield {"event": "progress", "data": data}

                    # Close stream when processing is complete
                    parsed = json.loads(data)
                    if parsed.get("stage") in ("ready", "error"):
                        break
        finally:
            await pubsub.unsubscribe(f"lexintel:case:{case_id}:progress")
            await redis.close()

    return EventSourceResponse(event_generator())
```

#### `backend/requirements.txt` (add dependency)

```
sse-starlette>=1.6.0
redis>=5.0.0  # Ensure async support
```

## Frontend Implementation

### New Files

#### `frontend/hooks/use-case-progress.ts`

```typescript
import { useState, useEffect, useCallback } from 'react'

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

export function useCaseProgress(caseId: string | null) {
  const [progress, setProgress] = useState<ProgressEvent | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [isComplete, setIsComplete] = useState(false)

  useEffect(() => {
    if (!caseId) return

    const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
    const token = localStorage.getItem('access_token')

    const eventSource = new EventSource(
      `${API_URL}/cases/${caseId}/progress?token=${token}`
    )

    eventSource.onopen = () => setIsConnected(true)

    eventSource.addEventListener('progress', (event) => {
      const data: ProgressEvent = JSON.parse(event.data)
      setProgress(data)

      if (data.stage === 'ready' || data.stage === 'error') {
        setIsComplete(true)
        eventSource.close()
      }
    })

    eventSource.onerror = () => {
      setIsConnected(false)
      eventSource.close()
    }

    return () => eventSource.close()
  }, [caseId])

  return { progress, isConnected, isComplete }
}
```

#### `frontend/components/processing-stepper.tsx`

```typescript
'use client'

import { Check, Loader2, Circle, AlertCircle, RotateCw } from 'lucide-react'
import { ProgressEvent } from '@/hooks/use-case-progress'
import { cn } from '@/lib/utils'

const STAGES = [
  { key: 'uploaded', label: 'Uploaded', description: 'Document received' },
  { key: 'downloading', label: 'Downloading', description: 'Fetching from storage' },
  { key: 'chunking', label: 'Chunking', description: 'Splitting document' },
  { key: 'embedding', label: 'Embedding', description: 'Generating vectors' },
  { key: 'indexing', label: 'Indexing', description: 'Storing in vector DB' },
  { key: 'storing', label: 'Storing', description: 'Saving metadata' },
  { key: 'ready', label: 'Ready', description: 'Ready for analysis' },
]

interface ProcessingStepperProps {
  progress: ProgressEvent | null
}

export default function ProcessingStepper({ progress }: ProcessingStepperProps) {
  const currentStep = progress?.step || 1
  const isError = progress?.stage === 'error'
  const isRetrying = progress?.stage === 'retrying'

  return (
    <div className="space-y-4">
      {STAGES.map((stage, index) => {
        const stepNumber = index + 1
        const isCompleted = stepNumber < currentStep
        const isCurrent = stepNumber === currentStep
        const isPending = stepNumber > currentStep

        return (
          <div key={stage.key} className="flex gap-4">
            {/* Icon */}
            <div className="flex flex-col items-center">
              <div
                className={cn(
                  'w-8 h-8 rounded-full flex items-center justify-center border-2',
                  isCompleted && 'bg-green-500 border-green-500',
                  isCurrent && !isError && 'bg-blue-500 border-blue-500',
                  isCurrent && isError && 'bg-red-500 border-red-500',
                  isPending && 'bg-white border-gray-300'
                )}
              >
                {isCompleted && <Check className="w-4 h-4 text-white" />}
                {isCurrent && !isError && !isRetrying && (
                  <Loader2 className="w-4 h-4 text-white animate-spin" />
                )}
                {isCurrent && isRetrying && (
                  <RotateCw className="w-4 h-4 text-white animate-spin" />
                )}
                {isCurrent && isError && (
                  <AlertCircle className="w-4 h-4 text-white" />
                )}
                {isPending && <Circle className="w-4 h-4 text-gray-400" />}
              </div>
              {index < STAGES.length - 1 && (
                <div
                  className={cn(
                    'w-0.5 h-8 mt-1',
                    isCompleted ? 'bg-green-500' : 'bg-gray-200'
                  )}
                />
              )}
            </div>

            {/* Content */}
            <div className="flex-1 pb-4">
              <div className="flex items-center justify-between">
                <h4
                  className={cn(
                    'font-medium',
                    isCompleted && 'text-green-700',
                    isCurrent && !isError && 'text-blue-700',
                    isCurrent && isError && 'text-red-700',
                    isPending && 'text-gray-400'
                  )}
                >
                  {stage.label}
                </h4>
                {isCurrent && progress?.progress > 0 && progress.progress < 100 && (
                  <span className="text-sm font-medium text-blue-600">
                    {progress.progress}%
                  </span>
                )}
              </div>

              <p className="text-sm text-gray-500">{stage.description}</p>

              {/* Progress bar for current stage */}
              {isCurrent && progress?.progress > 0 && (
                <div className="mt-2">
                  <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                    <div
                      className={cn(
                        'h-full transition-all duration-300',
                        isError ? 'bg-red-500' : 'bg-blue-500'
                      )}
                      style={{ width: `${progress.progress}%` }}
                    />
                  </div>
                  {progress.detail && (
                    <p className="text-xs text-gray-500 mt-1">{progress.detail}</p>
                  )}
                </div>
              )}

              {/* Error message */}
              {isCurrent && isError && progress?.error && (
                <p className="text-sm text-red-600 mt-1">{progress.error}</p>
              )}

              {/* Retry info */}
              {isCurrent && isRetrying && progress?.retry_attempt && (
                <p className="text-sm text-orange-600 mt-1">
                  Retry attempt {progress.retry_attempt}/3
                </p>
              )}
            </div>
          </div>
        )
      })}
    </div>
  )
}
```

#### `frontend/components/processing-modal.tsx`

```typescript
'use client'

import { X, FileText } from 'lucide-react'
import { Button } from '@/components/ui/button'
import ProcessingStepper from './processing-stepper'
import { useCaseProgress } from '@/hooks/use-case-progress'
import { useRouter } from 'next/navigation'

interface ProcessingModalProps {
  isOpen: boolean
  onClose: () => void
  caseId: string | null
  fileName: string
}

export default function ProcessingModal({
  isOpen,
  onClose,
  caseId,
  fileName,
}: ProcessingModalProps) {
  const router = useRouter()
  const { progress, isComplete } = useCaseProgress(caseId)

  if (!isOpen) return null

  const isReady = progress?.stage === 'ready'
  const isError = progress?.stage === 'error'

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/50"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative bg-white rounded-2xl shadow-xl w-full max-w-lg mx-4 p-6">
        {/* Close button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-gray-400 hover:text-gray-600"
        >
          <X className="w-5 h-5" />
        </button>

        {/* Header */}
        <div className="text-center mb-6">
          <div className="inline-flex items-center justify-center w-12 h-12 bg-blue-100 rounded-full mb-3">
            <FileText className="w-6 h-6 text-blue-600" />
          </div>
          <h3 className="text-lg font-semibold text-gray-900 truncate px-8">
            {fileName}
          </h3>
          <p className="text-sm text-gray-500">
            {isReady
              ? 'Processing complete!'
              : isError
              ? 'Processing failed'
              : 'Processing your document...'}
          </p>
        </div>

        {/* Stepper */}
        <div className="max-h-96 overflow-y-auto px-2">
          <ProcessingStepper progress={progress} />
        </div>

        {/* Footer */}
        <div className="mt-6 pt-4 border-t border-gray-200">
          <p className="text-xs text-gray-500 text-center mb-4">
            You can close this modal. We'll notify you when processing is complete.
          </p>
          <div className="flex gap-3">
            <Button variant="outline" onClick={onClose} className="flex-1">
              Close
            </Button>
            <Button
              onClick={() => router.push(`/cases/${caseId}`)}
              className="flex-1"
              disabled={!caseId}
            >
              {isReady ? 'Open Case' : 'View Case'}
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}
```

#### `frontend/components/case-card.tsx`

```typescript
'use client'

import { FileText, Trash2, ExternalLink, RotateCw } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useCaseProgress } from '@/hooks/use-case-progress'
import { cn } from '@/lib/utils'
import { formatDistanceToNow } from 'date-fns'

interface CaseCardProps {
  id: string
  name: string
  status: string
  fileType: string
  fileSize?: number
  chunkCount?: number
  createdAt: string
  onView: (id: string) => void
  onDelete: (id: string) => void
  onViewProgress: (id: string) => void
}

export default function CaseCard({
  id,
  name,
  status,
  fileType,
  fileSize,
  chunkCount,
  createdAt,
  onView,
  onDelete,
  onViewProgress,
}: CaseCardProps) {
  const isProcessing = status === 'processing'
  const { progress } = useCaseProgress(isProcessing ? id : null)

  const statusConfig = {
    processing: { label: 'Processing', color: 'bg-blue-100 text-blue-700' },
    ready: { label: 'Ready', color: 'bg-green-100 text-green-700' },
    error: { label: 'Error', color: 'bg-red-100 text-red-700' },
  }

  const currentStatus = statusConfig[status as keyof typeof statusConfig] || statusConfig.processing

  return (
    <div className="bg-white border border-gray-200 rounded-xl p-4 hover:shadow-md transition-shadow">
      {/* Header */}
      <div className="flex items-start justify-between gap-2 mb-3">
        <div className="flex items-center gap-2 min-w-0">
          <FileText className="w-5 h-5 text-gray-400 flex-shrink-0" />
          <h4 className="font-medium text-gray-900 truncate">{name}</h4>
        </div>
        <span
          className={cn(
            'px-2 py-1 text-xs font-medium rounded-full flex-shrink-0',
            currentStatus.color
          )}
        >
          {currentStatus.label}
        </span>
      </div>

      {/* Progress (if processing) */}
      {isProcessing && progress && (
        <div className="mb-3">
          <div className="flex items-center justify-between text-sm mb-1">
            <span className="text-gray-600">{progress.message}</span>
            {progress.progress > 0 && (
              <span className="text-blue-600 font-medium">{progress.progress}%</span>
            )}
          </div>
          <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-500 transition-all duration-300"
              style={{ width: `${progress.progress || 10}%` }}
            />
          </div>
        </div>
      )}

      {/* Meta info (if ready) */}
      {status === 'ready' && (
        <div className="text-sm text-gray-500 mb-3">
          {chunkCount && <span>{chunkCount} chunks</span>}
          {chunkCount && fileType && <span> • </span>}
          {fileType && <span>{fileType.toUpperCase()}</span>}
        </div>
      )}

      {/* Error info */}
      {status === 'error' && progress?.error && (
        <p className="text-sm text-red-600 mb-3 line-clamp-2">{progress.error}</p>
      )}

      {/* Footer */}
      <div className="flex items-center justify-between pt-2 border-t border-gray-100">
        <span className="text-xs text-gray-400">
          {formatDistanceToNow(new Date(createdAt), { addSuffix: true })}
        </span>
        <div className="flex gap-1">
          {isProcessing && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => onViewProgress(id)}
            >
              View Progress
            </Button>
          )}
          {status === 'ready' && (
            <Button variant="ghost" size="sm" onClick={() => onView(id)}>
              <ExternalLink className="w-4 h-4 mr-1" />
              Open
            </Button>
          )}
          {status === 'error' && (
            <Button variant="ghost" size="sm" onClick={() => onViewProgress(id)}>
              <RotateCw className="w-4 h-4 mr-1" />
              Retry
            </Button>
          )}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onDelete(id)}
            className="text-red-600 hover:text-red-700 hover:bg-red-50"
          >
            <Trash2 className="w-4 h-4" />
          </Button>
        </div>
      </div>
    </div>
  )
}
```

#### `frontend/components/cases-list.tsx`

```typescript
'use client'

import { useQuery } from '@tanstack/react-query'
import { useRouter } from 'next/navigation'
import CaseCard from './case-card'
import apiClient from '@/lib/api'

interface Case {
  id: string
  name: string
  status: string
  file_type: string
  created_at: string
  chunk_count?: number
}

interface CasesListProps {
  onViewProgress: (caseId: string, fileName: string) => void
}

export default function CasesList({ onViewProgress }: CasesListProps) {
  const router = useRouter()

  const { data: cases, isLoading } = useQuery({
    queryKey: ['cases'],
    queryFn: async () => {
      const response = await apiClient.get('/cases')
      return response.data as Case[]
    },
    refetchInterval: 10000, // Refetch every 10s to catch status changes
  })

  const handleView = (id: string) => {
    router.push(`/cases/${id}`)
  }

  const handleDelete = async (id: string) => {
    if (confirm('Are you sure you want to delete this case?')) {
      await apiClient.delete(`/cases/${id}`)
      // Refetch will happen automatically
    }
  }

  if (isLoading) {
    return (
      <div className="bg-white rounded-2xl shadow-lg p-8">
        <div className="animate-pulse space-y-4">
          <div className="h-6 bg-gray-200 rounded w-32" />
          <div className="h-24 bg-gray-200 rounded" />
        </div>
      </div>
    )
  }

  if (!cases?.length) {
    return null // Don't show section if no cases
  }

  const recentCases = cases.slice(0, 3)
  const totalCases = cases.length

  return (
    <div className="bg-white rounded-2xl shadow-lg p-8">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-bold text-gray-900">My Cases</h2>
        {totalCases > 3 && (
          <button
            onClick={() => router.push('/cases')}
            className="text-sm text-blue-600 hover:underline font-medium"
          >
            View All →
          </button>
        )}
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {recentCases.map((caseItem) => (
          <CaseCard
            key={caseItem.id}
            id={caseItem.id}
            name={caseItem.name}
            status={caseItem.status}
            fileType={caseItem.file_type}
            chunkCount={caseItem.chunk_count}
            createdAt={caseItem.created_at}
            onView={handleView}
            onDelete={handleDelete}
            onViewProgress={(id) => onViewProgress(id, caseItem.name)}
          />
        ))}
      </div>

      <p className="text-sm text-gray-500 mt-4">
        Showing {recentCases.length} most recent • {totalCases} total cases
      </p>
    </div>
  )
}
```

### Modified Files

#### `frontend/app/dashboard/page.tsx` (key changes)

Add state for modal and integrate new components:

```typescript
// Add imports
import ProcessingModal from '@/components/processing-modal'
import CasesList from '@/components/cases-list'
import { useToast } from '@/components/ui/use-toast'

// Add state
const [processingModal, setProcessingModal] = useState<{
  isOpen: boolean
  caseId: string | null
  fileName: string
}>({ isOpen: false, caseId: null, fileName: '' })

// Modify upload mutation onSuccess
onSuccess: (data) => {
  setProcessingModal({
    isOpen: true,
    caseId: data.id,
    fileName: selectedFile?.name || 'Document',
  })
  // Reset form
  setCaseName('')
  setSelectedFile(null)
}

// Add to JSX after the main upload card
<ProcessingModal
  isOpen={processingModal.isOpen}
  onClose={() => setProcessingModal(prev => ({ ...prev, isOpen: false }))}
  caseId={processingModal.caseId}
  fileName={processingModal.fileName}
/>

<CasesList
  onViewProgress={(caseId, fileName) =>
    setProcessingModal({ isOpen: true, caseId, fileName })
  }
/>
```

## Toast Notifications

Use existing toast system or add shadcn/ui toast. Trigger when SSE receives `ready` stage while modal is closed:

```typescript
// In CasesList or a global listener
useEffect(() => {
  if (progress?.stage === 'ready' && !modalOpen) {
    toast({
      title: `${caseName} is ready!`,
      description: `${progress.detail}`,
      action: <Button onClick={() => router.push(`/cases/${caseId}`)}>View</Button>,
    })
  }
}, [progress?.stage])
```

## Dependencies

### Backend
```
sse-starlette>=1.6.0
redis>=5.0.0
```

### Frontend
```
date-fns  # For relative time formatting (if not installed)
```

## File Summary

| File | Action | Description |
|------|--------|-------------|
| `backend/services/progress.py` | CREATE | Redis pub/sub publisher |
| `backend/tasks.py` | MODIFY | Add publish_progress calls |
| `backend/main.py` | MODIFY | Add SSE endpoint |
| `backend/requirements.txt` | MODIFY | Add sse-starlette |
| `frontend/hooks/use-case-progress.ts` | CREATE | SSE subscription hook |
| `frontend/components/processing-stepper.tsx` | CREATE | Vertical stepper UI |
| `frontend/components/processing-modal.tsx` | CREATE | Modal wrapper |
| `frontend/components/case-card.tsx` | CREATE | Case card component |
| `frontend/components/cases-list.tsx` | CREATE | My Cases section |
| `frontend/app/dashboard/page.tsx` | MODIFY | Integrate modal + cases list |

## Testing Plan

1. Upload a document and verify modal opens with stepper
2. Verify each stage updates in real-time
3. Close modal mid-processing, verify case appears in My Cases
4. Verify toast appears when processing completes (modal closed)
5. Test error state with invalid document
6. Test retry visibility with rate-limited API
7. Verify "Open Case" works when ready
