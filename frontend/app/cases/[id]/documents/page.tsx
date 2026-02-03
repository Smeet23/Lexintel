'use client'

import React from 'react'
import { useParams } from 'next/navigation'
import useDocuments from '@/hooks/useDocuments'
import DocumentUploader from '@/components/document-uploader'
import DocumentManager from '@/components/document-manager'
import { Alert } from '@/components/ui/alert'
import { AlertCircle } from 'lucide-react'

export default function DocumentsPage() {
  const params = useParams()
  const caseId = params.id as string

  const { documents, isLoading, error, refetch, deleteDocument } = useDocuments(caseId)

  return (
    <div className="container mx-auto py-8 px-4 max-w-4xl">
      <div className="mb-8">
        <DocumentUploader caseId={caseId} onUploadComplete={refetch} />
      </div>

      <div className="mt-12">
        <h2 className="text-2xl font-bold mb-6">Documents</h2>

        {error && (
          <Alert variant="destructive" className="flex items-start gap-3 mb-6">
            <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
            <div>Failed to load documents. Please try again.</div>
          </Alert>
        )}

        {isLoading ? (
          <div className="flex items-center justify-center py-12">
            <div className="w-8 h-8 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin" />
          </div>
        ) : (
          <DocumentManager caseId={caseId} documents={documents} onDelete={deleteDocument} />
        )}
      </div>
    </div>
  )
}
