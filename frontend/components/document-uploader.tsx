'use client'

import React, { useRef, useState } from 'react'
import { Upload, AlertCircle, CheckCircle } from 'lucide-react'
import { useMutation } from '@tanstack/react-query'
import api from '@/lib/api'
import { Button } from '@/components/ui/button'
import { Alert } from '@/components/ui/alert'

interface DocumentUploaderProps {
  caseId: string
  onUploadComplete?: (document: UploadedDocument) => void
}

interface UploadedDocument {
  id: string
  filename: string
  status: string
}

const MAX_FILE_SIZE = 50 * 1024 * 1024 // 50MB
const ALLOWED_TYPES = [
  'application/pdf',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  'text/plain',
]

export default function DocumentUploader({ caseId, onUploadComplete }: DocumentUploaderProps) {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  const [dragActive, setDragActive] = useState(false)

  const uploadMutation = useMutation({
    mutationFn: async (file: File) => {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('case_id', caseId)

      const response = await api.post(`/cases/${caseId}/upload-document`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      return response.data
    },
    onSuccess: (data) => {
      setError(null)
      setSuccess('Upload successful')
      onUploadComplete?.(data)
      setTimeout(() => setSuccess(null), 3000)
    },
    onError: (error: any) => {
      setSuccess(null)
      setError(error.response?.data?.message || 'Upload failed')
    },
  })

  const validateFile = (file: File): string | null => {
    if (!ALLOWED_TYPES.includes(file.type)) {
      return 'Invalid file type. Please upload PDF, DOCX, or TXT files.'
    }

    if (file.size > MAX_FILE_SIZE) {
      return 'File too large. Maximum size is 50MB.'
    }

    return null
  }

  const handleFileSelect = (files: FileList | null) => {
    if (!files) return

    setError(null)
    setSuccess(null)

    Array.from(files).forEach((file) => {
      const validationError = validateFile(file)
      if (validationError) {
        setError(validationError)
        return
      }

      uploadMutation.mutate(file)
    })
  }

  const handleDrag = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()

    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    handleFileSelect(e.dataTransfer.files)
  }

  const handleClick = () => {
    fileInputRef.current?.click()
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    handleFileSelect(e.target.files)
  }

  return (
    <div className="w-full space-y-4">
      <h2 className="text-2xl font-bold">Upload Documents</h2>

      <div
        className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
          dragActive
            ? 'border-blue-500 bg-blue-50'
            : 'border-gray-300 hover:border-gray-400'
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={handleClick}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".pdf,.docx,.txt,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,text/plain"
          onChange={handleInputChange}
          className="hidden"
          aria-label="file-input"
        />

        <Upload className="w-12 h-12 mx-auto mb-4 text-gray-400" />
        <p className="text-lg font-medium mb-2">Drag and drop files here</p>
        <p className="text-sm text-gray-500 mb-4">or</p>
        <Button onClick={handleClick} type="button">
          Select Files
        </Button>
        <p className="text-xs text-gray-500 mt-4">
          Supported formats: PDF, DOCX, TXT (Max 50MB per file)
        </p>
      </div>

      {uploadMutation.isPending && (
        <div className="flex items-center gap-2 text-blue-600">
          <div className="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
          <span>Uploading...</span>
        </div>
      )}

      {error && (
        <Alert variant="destructive" className="flex items-start gap-3">
          <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
          <div>{error}</div>
        </Alert>
      )}

      {success && (
        <Alert className="flex items-start gap-3 bg-green-50 border-green-200">
          <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
          <div className="text-green-800">{success}</div>
        </Alert>
      )}
    </div>
  )
}
