'use client'

import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useRouter } from 'next/navigation'
import { FolderOpen } from 'lucide-react'
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
  limit?: number
}

export default function CasesList({ onViewProgress, limit = 6 }: CasesListProps) {
  const router = useRouter()
  const queryClient = useQueryClient()

  const { data: cases, isLoading, error } = useQuery({
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
    if (!confirm('Are you sure you want to delete this case?')) {
      return
    }

    try {
      await apiClient.delete(`/cases/${id}`)
      // Invalidate and refetch
      queryClient.invalidateQueries({ queryKey: ['cases'] })
    } catch (error) {
      console.error('Failed to delete case:', error)
      alert('Failed to delete case. Please try again.')
    }
  }

  if (isLoading) {
    return (
      <div className="bg-white rounded-2xl shadow-lg p-8">
        <div className="animate-pulse space-y-4">
          <div className="h-6 bg-gray-200 rounded w-32" />
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-40 bg-gray-100 rounded-xl" />
            ))}
          </div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-white rounded-2xl shadow-lg p-8">
        <div className="text-center py-8">
          <p className="text-red-600">Failed to load cases</p>
          <button
            onClick={() => queryClient.invalidateQueries({ queryKey: ['cases'] })}
            className="mt-2 text-blue-600 hover:underline"
          >
            Try again
          </button>
        </div>
      </div>
    )
  }

  if (!cases?.length) {
    return (
      <div className="bg-white rounded-2xl shadow-lg p-8">
        <h2 className="text-xl font-bold text-gray-900 mb-6">My Cases</h2>
        <div className="text-center py-12">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-gray-100 rounded-full mb-4">
            <FolderOpen className="w-8 h-8 text-gray-400" />
          </div>
          <h3 className="text-lg font-medium text-gray-900 mb-2">No cases yet</h3>
          <p className="text-gray-500">Upload your first document to get started</p>
        </div>
      </div>
    )
  }

  // Sort by created_at descending (most recent first)
  const sortedCases = [...cases].sort(
    (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
  )

  const displayedCases = sortedCases.slice(0, limit)
  const totalCases = cases.length
  const hasMore = totalCases > limit

  return (
    <div className="bg-white rounded-2xl shadow-lg p-8">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-bold text-gray-900">My Cases</h2>
        {hasMore && (
          <button
            onClick={() => router.push('/cases')}
            className="text-sm text-blue-600 hover:text-blue-700 hover:underline font-medium"
          >
            View All ({totalCases})
          </button>
        )}
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {displayedCases.map((caseItem) => (
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
            onViewProgress={onViewProgress}
          />
        ))}
      </div>

      <p className="text-sm text-gray-500 mt-6 text-center">
        Showing {displayedCases.length} of {totalCases} case{totalCases !== 1 ? 's' : ''}
      </p>
    </div>
  )
}
