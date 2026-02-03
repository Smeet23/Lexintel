'use client'

import React, { useState } from 'react'
import { Trash2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { formatFileSize, formatDate, getStatusColor } from '@/lib/document-utils'

export interface DocumentItem {
  id: string
  filename: string
  size: number
  status: 'pending' | 'processing' | 'ready' | 'error'
  created_at: string
}

interface DocumentManagerProps {
  caseId: string
  documents: DocumentItem[]
  onDelete?: (documentId: string) => void
}

export default function DocumentManager({
  caseId,
  documents,
  onDelete,
}: DocumentManagerProps) {
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null)

  const handleDeleteClick = (documentId: string) => {
    setDeleteConfirmId(documentId)
  }

  const confirmDelete = () => {
    if (deleteConfirmId && onDelete) {
      onDelete(deleteConfirmId)
      setDeleteConfirmId(null)
    }
  }

  const cancelDelete = () => {
    setDeleteConfirmId(null)
  }

  if (documents.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500">
        <p>No documents uploaded yet</p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="border-b">
            <tr>
              <th className="text-left py-2 px-4">Filename</th>
              <th className="text-left py-2 px-4">Size</th>
              <th className="text-left py-2 px-4">Status</th>
              <th className="text-left py-2 px-4">Uploaded</th>
              <th className="text-left py-2 px-4">Actions</th>
            </tr>
          </thead>
          <tbody>
            {documents.map((doc) => (
              <tr key={doc.id} className="border-b hover:bg-gray-50">
                <td className="py-2 px-4">{doc.filename}</td>
                <td className="py-2 px-4">{formatFileSize(doc.size)}</td>
                <td className="py-2 px-4">
                  <Badge variant={getStatusColor(doc.status)}>{doc.status}</Badge>
                </td>
                <td className="py-2 px-4">{formatDate(doc.created_at)}</td>
                <td className="py-2 px-4">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleDeleteClick(doc.id)}
                    disabled={deleteConfirmId === doc.id}
                  >
                    <Trash2 className="w-4 h-4" />
                  </Button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {deleteConfirmId && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 shadow-lg max-w-sm">
            <h3 className="text-lg font-semibold mb-4">Confirm deletion</h3>
            <p className="text-gray-600 mb-6">
              Are you sure you want to delete this document? This action cannot be undone.
            </p>
            <div className="flex gap-3 justify-end">
              <Button variant="outline" onClick={cancelDelete}>
                Cancel
              </Button>
              <Button variant="destructive" onClick={confirmDelete}>
                Delete
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
