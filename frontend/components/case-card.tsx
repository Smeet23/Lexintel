'use client'

import { FileText, Trash2, ExternalLink, RotateCw, Eye } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useCaseProgress } from '@/hooks/use-case-progress'
import { cn } from '@/lib/utils'

interface CaseCardProps {
  id: string
  name: string
  status: string
  fileType: string
  chunkCount?: number
  createdAt: string
  onView: (id: string) => void
  onDelete: (id: string) => void
  onViewProgress: (id: string, name: string) => void
}

function formatRelativeTime(dateString: string): string {
  const date = new Date(dateString)
  const now = new Date()
  const diffInSeconds = Math.floor((now.getTime() - date.getTime()) / 1000)

  if (diffInSeconds < 60) return 'Just now'
  if (diffInSeconds < 3600) {
    const minutes = Math.floor(diffInSeconds / 60)
    return `${minutes} minute${minutes > 1 ? 's' : ''} ago`
  }
  if (diffInSeconds < 86400) {
    const hours = Math.floor(diffInSeconds / 3600)
    return `${hours} hour${hours > 1 ? 's' : ''} ago`
  }
  const days = Math.floor(diffInSeconds / 86400)
  if (days < 7) {
    return `${days} day${days > 1 ? 's' : ''} ago`
  }
  return date.toLocaleDateString()
}

export default function CaseCard({
  id,
  name,
  status,
  fileType,
  chunkCount,
  createdAt,
  onView,
  onDelete,
  onViewProgress,
}: CaseCardProps) {
  const isProcessing = status === 'processing'
  const { progress } = useCaseProgress(isProcessing ? id : null)

  // Use progress stage if available, otherwise use status
  const currentStage = progress?.stage || status

  const statusConfig: Record<string, { label: string; color: string }> = {
    processing: { label: 'Processing', color: 'bg-blue-100 text-blue-700' },
    ready: { label: 'Ready', color: 'bg-green-100 text-green-700' },
    error: { label: 'Error', color: 'bg-red-100 text-red-700' },
    retrying: { label: 'Retrying', color: 'bg-orange-100 text-orange-700' },
  }

  const currentStatus = statusConfig[currentStage] || statusConfig.processing

  return (
    <div className="bg-white border border-gray-200 rounded-xl p-5 hover:shadow-lg hover:border-gray-300 transition-all duration-200">
      {/* Header */}
      <div className="flex items-start justify-between gap-3 mb-3">
        <div className="flex items-center gap-3 min-w-0">
          <div className="flex-shrink-0 w-10 h-10 bg-gray-100 rounded-lg flex items-center justify-center">
            <FileText className="w-5 h-5 text-gray-500" />
          </div>
          <div className="min-w-0">
            <h4 className="font-semibold text-gray-900 truncate" title={name}>
              {name}
            </h4>
            <p className="text-xs text-gray-500">
              {fileType?.toUpperCase() || 'PDF'}
            </p>
          </div>
        </div>
        <span
          className={cn(
            'px-2.5 py-1 text-xs font-medium rounded-full flex-shrink-0',
            currentStatus.color
          )}
        >
          {currentStatus.label}
        </span>
      </div>

      {/* Progress (if processing) */}
      {(isProcessing || currentStage === 'retrying') && (
        <div className="mb-4">
          <div className="flex items-center justify-between text-sm mb-1.5">
            <span className="text-gray-600 truncate">
              {progress?.message || 'Starting...'}
            </span>
            {progress && progress.progress > 0 && (
              <span className="text-blue-600 font-semibold ml-2">
                {progress.progress}%
              </span>
            )}
          </div>
          <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
            <div
              className={cn(
                'h-full transition-all duration-500 ease-out rounded-full',
                currentStage === 'retrying' ? 'bg-orange-500' : 'bg-blue-500'
              )}
              style={{ width: `${progress?.progress || 5}%` }}
            />
          </div>
          {progress?.detail && (
            <p className="text-xs text-gray-500 mt-1">{progress.detail}</p>
          )}
        </div>
      )}

      {/* Meta info (if ready) */}
      {status === 'ready' && (
        <div className="text-sm text-gray-600 mb-4 flex items-center gap-2">
          {chunkCount !== undefined && chunkCount > 0 && (
            <>
              <span className="font-medium">{chunkCount}</span>
              <span>chunks</span>
              <span className="text-gray-300">|</span>
            </>
          )}
          <span>Ready for analysis</span>
        </div>
      )}

      {/* Error info */}
      {status === 'error' && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-sm text-red-700 line-clamp-2">
            {progress?.error || 'Processing failed. Please try again.'}
          </p>
        </div>
      )}

      {/* Footer */}
      <div className="flex items-center justify-between pt-3 border-t border-gray-100">
        <span className="text-xs text-gray-400">
          {formatRelativeTime(createdAt)}
        </span>
        <div className="flex gap-1">
          {isProcessing && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => onViewProgress(id, name)}
              className="text-blue-600 hover:text-blue-700 hover:bg-blue-50"
            >
              <Eye className="w-4 h-4 mr-1" />
              Progress
            </Button>
          )}
          {status === 'ready' && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => onView(id)}
              className="text-green-600 hover:text-green-700 hover:bg-green-50"
            >
              <ExternalLink className="w-4 h-4 mr-1" />
              Open
            </Button>
          )}
          {status === 'error' && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => onViewProgress(id, name)}
              className="text-orange-600 hover:text-orange-700 hover:bg-orange-50"
            >
              <RotateCw className="w-4 h-4 mr-1" />
              Details
            </Button>
          )}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onDelete(id)}
            className="text-gray-400 hover:text-red-600 hover:bg-red-50"
          >
            <Trash2 className="w-4 h-4" />
          </Button>
        </div>
      </div>
    </div>
  )
}
