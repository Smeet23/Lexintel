import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import api from '@/lib/api'
import { DocumentItem } from '@/components/document-manager'

interface DocumentsResponse {
  documents: DocumentItem[]
}

export default function useDocuments(caseId: string) {
  const queryClient = useQueryClient()

  const {
    data: documentsData,
    isLoading,
    error,
    refetch,
  } = useQuery<DocumentsResponse>({
    queryKey: ['case-documents', caseId],
    queryFn: async () => {
      const response = await api.get(`/cases/${caseId}/documents`)
      return response.data
    },
    enabled: !!caseId,
    staleTime: 5000,
  })

  const deleteDocumentMutation = useMutation({
    mutationFn: async (documentId: string) => {
      await api.delete(`/cases/${caseId}/documents/${documentId}`)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['case-documents', caseId] })
    },
  })

  return {
    documents: documentsData?.documents || [],
    isLoading,
    error,
    refetch,
    deleteDocument: (documentId: string) => deleteDocumentMutation.mutateAsync(documentId),
    isDeleting: deleteDocumentMutation.isPending,
  }
}
