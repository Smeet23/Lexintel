'use client'

import { Check, Loader2, Circle, AlertCircle, RotateCw } from 'lucide-react'
import { cn } from '@/lib/utils'
import type { ProgressEvent } from '@/hooks/use-case-progress'

const STAGES = [
  { key: 'uploaded', label: 'Uploaded', description: 'Document received' },
  { key: 'downloading', label: 'Downloading', description: 'Fetching from storage' },
  { key: 'chunking', label: 'Chunking', description: 'Splitting document into chunks' },
  { key: 'embedding', label: 'Embedding', description: 'Generating vector embeddings' },
  { key: 'indexing', label: 'Indexing', description: 'Storing in vector database' },
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
    <div className="space-y-1">
      {STAGES.map((stage, index) => {
        const stepNumber = index + 1
        const isCompleted = stepNumber < currentStep || progress?.stage === 'ready'
        const isCurrent = stepNumber === currentStep && progress?.stage !== 'ready'
        const isPending = stepNumber > currentStep

        // For ready state, all steps are completed
        const isReadyStage = stage.key === 'ready' && progress?.stage === 'ready'

        return (
          <div key={stage.key} className="flex gap-4">
            {/* Icon column */}
            <div className="flex flex-col items-center">
              <div
                className={cn(
                  'w-8 h-8 rounded-full flex items-center justify-center border-2 transition-all duration-300',
                  (isCompleted || isReadyStage) && 'bg-green-500 border-green-500',
                  isCurrent && !isError && !isRetrying && 'bg-blue-500 border-blue-500',
                  isCurrent && isRetrying && 'bg-orange-500 border-orange-500',
                  isCurrent && isError && 'bg-red-500 border-red-500',
                  isPending && 'bg-white border-gray-300'
                )}
              >
                {(isCompleted || isReadyStage) && (
                  <Check className="w-4 h-4 text-white" />
                )}
                {isCurrent && !isError && !isRetrying && (
                  <Loader2 className="w-4 h-4 text-white animate-spin" />
                )}
                {isCurrent && isRetrying && (
                  <RotateCw className="w-4 h-4 text-white animate-spin" />
                )}
                {isCurrent && isError && (
                  <AlertCircle className="w-4 h-4 text-white" />
                )}
                {isPending && (
                  <Circle className="w-3 h-3 text-gray-400" />
                )}
              </div>
              {/* Connector line */}
              {index < STAGES.length - 1 && (
                <div
                  className={cn(
                    'w-0.5 h-12 transition-all duration-300',
                    isCompleted ? 'bg-green-500' : 'bg-gray-200'
                  )}
                />
              )}
            </div>

            {/* Content column */}
            <div className="flex-1 pb-6">
              <div className="flex items-center justify-between">
                <h4
                  className={cn(
                    'font-medium transition-colors duration-300',
                    (isCompleted || isReadyStage) && 'text-green-700',
                    isCurrent && !isError && !isRetrying && 'text-blue-700',
                    isCurrent && isRetrying && 'text-orange-700',
                    isCurrent && isError && 'text-red-700',
                    isPending && 'text-gray-400'
                  )}
                >
                  {stage.label}
                </h4>
                {isCurrent && progress && progress.progress > 0 && progress.progress < 100 && (
                  <span className="text-sm font-semibold text-blue-600">
                    {progress.progress}%
                  </span>
                )}
              </div>

              <p className={cn(
                'text-sm mt-0.5',
                (isCompleted || isReadyStage || isCurrent) ? 'text-gray-600' : 'text-gray-400'
              )}>
                {isCurrent && progress?.message ? progress.message : stage.description}
              </p>

              {/* Progress bar for current stage */}
              {isCurrent && progress && progress.progress > 0 && (
                <div className="mt-2">
                  <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                    <div
                      className={cn(
                        'h-full transition-all duration-500 ease-out rounded-full',
                        isError ? 'bg-red-500' : isRetrying ? 'bg-orange-500' : 'bg-blue-500'
                      )}
                      style={{ width: `${progress.progress}%` }}
                    />
                  </div>
                  {progress.detail && (
                    <p className="text-xs text-gray-500 mt-1">{progress.detail}</p>
                  )}
                </div>
              )}

              {/* Completed detail */}
              {isCompleted && stage.key === progress?.stage && progress?.detail && (
                <p className="text-xs text-green-600 mt-1">{progress.detail}</p>
              )}

              {/* Error message */}
              {isCurrent && isError && progress?.error && (
                <div className="mt-2 p-2 bg-red-50 border border-red-200 rounded-lg">
                  <p className="text-sm text-red-700">{progress.error}</p>
                </div>
              )}

              {/* Retry info */}
              {isCurrent && isRetrying && progress?.retry_attempt && (
                <div className="mt-2 p-2 bg-orange-50 border border-orange-200 rounded-lg">
                  <p className="text-sm text-orange-700">
                    Retrying... (attempt {progress.retry_attempt}/3)
                  </p>
                  {progress.error && (
                    <p className="text-xs text-orange-600 mt-1">{progress.error}</p>
                  )}
                </div>
              )}
            </div>
          </div>
        )
      })}
    </div>
  )
}
