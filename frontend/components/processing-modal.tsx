'use client'

import { useEffect } from 'react'
import { X, FileText, CheckCircle, AlertCircle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import ProcessingStepper from './processing-stepper'
import { useCaseProgress } from '@/hooks/use-case-progress'
import { useRouter } from 'next/navigation'
import { cn } from '@/lib/utils'

interface ProcessingModalProps {
  isOpen: boolean
  onClose: () => void
  caseId: string | null
  fileName: string
  onComplete?: () => void
}

export default function ProcessingModal({
  isOpen,
  onClose,
  caseId,
  fileName,
  onComplete,
}: ProcessingModalProps) {
  const router = useRouter()
  const { progress, isComplete, isConnected } = useCaseProgress(caseId)

  const isReady = progress?.stage === 'ready'
  const isError = progress?.stage === 'error'

  // Call onComplete when processing finishes
  useEffect(() => {
    if (isComplete && isReady && onComplete) {
      onComplete()
    }
  }, [isComplete, isReady, onComplete])

  // Handle escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose()
      }
    }
    window.addEventListener('keydown', handleEscape)
    return () => window.removeEventListener('keydown', handleEscape)
  }, [isOpen, onClose])

  // Prevent body scroll when modal is open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden'
    } else {
      document.body.style.overflow = ''
    }
    return () => {
      document.body.style.overflow = ''
    }
  }, [isOpen])

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/50 backdrop-blur-sm transition-opacity"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative bg-white rounded-2xl shadow-2xl w-full max-w-lg max-h-[90vh] flex flex-col animate-in fade-in zoom-in-95 duration-200">
        {/* Close button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-1 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-full transition-colors z-10"
          aria-label="Close modal"
        >
          <X className="w-5 h-5" />
        </button>

        {/* Header */}
        <div className="text-center pt-6 pb-4 px-6 border-b border-gray-100">
          <div
            className={cn(
              'inline-flex items-center justify-center w-14 h-14 rounded-full mb-4 transition-colors',
              isReady && 'bg-green-100',
              isError && 'bg-red-100',
              !isReady && !isError && 'bg-blue-100'
            )}
          >
            {isReady && <CheckCircle className="w-7 h-7 text-green-600" />}
            {isError && <AlertCircle className="w-7 h-7 text-red-600" />}
            {!isReady && !isError && <FileText className="w-7 h-7 text-blue-600" />}
          </div>
          <h3 className="text-lg font-semibold text-gray-900 truncate px-8" title={fileName}>
            {fileName}
          </h3>
          <p className={cn(
            'text-sm mt-1',
            isReady && 'text-green-600',
            isError && 'text-red-600',
            !isReady && !isError && 'text-gray-500'
          )}>
            {isReady
              ? 'Processing complete!'
              : isError
              ? 'Processing failed'
              : 'Processing your document...'}
          </p>
          {!isConnected && !isComplete && (
            <p className="text-xs text-orange-500 mt-1">Connecting...</p>
          )}
        </div>

        {/* Stepper */}
        <div className="flex-1 overflow-y-auto px-6 py-4">
          <ProcessingStepper progress={progress} />
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-gray-100 bg-gray-50 rounded-b-2xl">
          {!isComplete && (
            <p className="text-xs text-gray-500 text-center mb-4">
              You can close this modal. We'll notify you when processing is complete.
            </p>
          )}
          <div className="flex gap-3">
            <Button
              variant="outline"
              onClick={onClose}
              className="flex-1"
            >
              {isComplete ? 'Close' : 'Close & Continue'}
            </Button>
            <Button
              onClick={() => router.push(`/cases/${caseId}`)}
              className={cn(
                'flex-1',
                isReady && 'bg-green-600 hover:bg-green-700',
                isError && 'bg-gray-400'
              )}
              disabled={!caseId || isError}
            >
              {isReady ? 'Open Case' : 'View Case'}
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}
