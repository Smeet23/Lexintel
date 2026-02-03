'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { useMutation } from '@tanstack/react-query'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { AlertCircle, CheckCircle, Lock } from 'lucide-react'
import DocumentUploader from '@/components/document-uploader'
import { formatFileSize } from '@/lib/document-utils'
import apiClient from '@/lib/api'

const MIN_CASE_NAME_LENGTH = 2
const MAX_CASE_NAME_LENGTH = 255

interface ValidationError {
  field: string
  message: string
}

export default function Dashboard() {
  const router = useRouter()
  const [caseName, setCaseName] = useState('')
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [errors, setErrors] = useState<Record<string, string>>({})

  const uploadMutation = useMutation({
    mutationFn: async (formData: FormData) => {
      const response = await apiClient.post('/cases', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })
      return response.data
    },
    onSuccess: (data) => {
      router.push(`/cases/${data.id}`)
    },
    onError: (err: any) => {
      setErrors({
        submit: err.response?.data?.detail || 'Failed to upload document. Please try again.',
      })
    },
  })

  const validateCaseName = (name: string): string | null => {
    if (!name.trim()) {
      return 'Please enter a case name'
    }
    if (name.length < MIN_CASE_NAME_LENGTH) {
      return `Case name must be at least ${MIN_CASE_NAME_LENGTH} characters`
    }
    if (name.length > MAX_CASE_NAME_LENGTH) {
      return `Case name is too long (maximum ${MAX_CASE_NAME_LENGTH} characters)`
    }
    return null
  }

  const handleCaseNameChange = (value: string) => {
    setCaseName(value)
    const error = validateCaseName(value)
    if (error) {
      setErrors((prev) => ({ ...prev, caseName: error }))
    } else {
      setErrors((prev) => {
        const newErrors = { ...prev }
        delete newErrors.caseName
        return newErrors
      })
    }
  }

  const handleFileSelect = (file: File) => {
    setSelectedFile(file)
    setErrors((prev) => {
      const newErrors = { ...prev }
      delete newErrors.file
      delete newErrors.submit
      return newErrors
    })
  }

  const validateForm = (): boolean => {
    const newErrors: Record<string, string> = {}

    const caseNameError = validateCaseName(caseName)
    if (caseNameError) {
      newErrors.caseName = caseNameError
    }

    if (!selectedFile) {
      newErrors.file = 'Please select a file'
    }

    setErrors(newErrors)
    return Object.keys(newErrors).length === 0
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()

    if (!validateForm()) {
      return
    }

    const formData = new FormData()
    formData.append('case_name', caseName)
    formData.append('file', selectedFile!)

    uploadMutation.mutate(formData)
  }

  const isFormValid =
    caseName.trim().length >= MIN_CASE_NAME_LENGTH &&
    caseName.length <= MAX_CASE_NAME_LENGTH &&
    !!selectedFile &&
    !uploadMutation.isPending

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 py-12 px-4">
      <div className="max-w-2xl mx-auto">
        {/* Header */}
        <div className="mb-12 text-center">
          <div className="inline-flex items-center justify-center w-14 h-14 bg-blue-100 rounded-full mb-4">
            <Lock className="w-7 h-7 text-blue-600" />
          </div>
          <h1 className="text-5xl font-bold text-gray-900 mb-3">LexIntel</h1>
          <p className="text-xl text-gray-600">Legal Document Analysis Platform</p>
        </div>

        {/* Main Card */}
        <div className="bg-white rounded-2xl shadow-lg p-8 md:p-10">
          <form onSubmit={handleSubmit} className="space-y-8">
            {/* Title */}
            <div>
              <h2 className="text-2xl font-bold text-gray-900 mb-2">Create New Case</h2>
              <p className="text-gray-600">Upload and analyze legal documents for your case</p>
            </div>

            {/* Case Name Field */}
            <div className="space-y-3">
              <label htmlFor="caseName" className="block text-sm font-semibold text-gray-900">
                Case Name
              </label>
              <Input
                id="caseName"
                type="text"
                value={caseName}
                onChange={(e) => handleCaseNameChange(e.target.value)}
                placeholder="e.g., Smith vs. Johnson LLC"
                disabled={uploadMutation.isPending}
                maxLength={MAX_CASE_NAME_LENGTH}
                className={`h-12 text-base transition-all ${
                  errors.caseName
                    ? 'border-2 border-red-500 focus:ring-red-200'
                    : 'border-2 border-gray-200 focus:border-blue-500'
                }`}
                aria-invalid={!!errors.caseName}
                aria-describedby={errors.caseName ? 'caseName-error' : undefined}
              />
              <div className="flex justify-between items-center">
                {errors.caseName && (
                  <span
                    id="caseName-error"
                    className="text-sm text-red-600 flex items-center gap-1 font-medium"
                  >
                    <AlertCircle className="w-4 h-4" />
                    {errors.caseName}
                  </span>
                )}
                <span
                  className={`text-xs font-medium ml-auto transition-colors ${
                    caseName.length > MAX_CASE_NAME_LENGTH * 0.9
                      ? 'text-orange-600'
                      : 'text-gray-500'
                  }`}
                >
                  {caseName.length}/{MAX_CASE_NAME_LENGTH}
                </span>
              </div>
            </div>

            {/* Divider */}
            <div className="border-t border-gray-200" />

            {/* Document Upload */}
            <div className="space-y-3">
              <label className="block text-sm font-semibold text-gray-900">Upload Document</label>
              <DocumentUploader caseId="temp" onUploadComplete={handleFileSelect} />
              {errors.file && (
                <div className="text-sm text-red-600 flex items-center gap-2 font-medium">
                  <AlertCircle className="w-4 h-4" />
                  {errors.file}
                </div>
              )}
            </div>

            {/* File Preview */}
            {selectedFile && (
              <div className="bg-gradient-to-r from-green-50 to-emerald-50 border-2 border-green-200 rounded-xl p-5 flex items-start gap-4">
                <div className="flex-shrink-0">
                  <CheckCircle className="w-6 h-6 text-green-600" />
                </div>
                <div className="flex-1">
                  <h4 className="font-semibold text-green-900">Document Ready</h4>
                  <div className="mt-2 space-y-1">
                    <p className="text-sm text-green-800">
                      <span className="font-medium">{selectedFile.name}</span>
                    </p>
                    <p className="text-xs text-green-700">
                      Size: <span className="font-medium">{formatFileSize(selectedFile.size)}</span>
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Error Message */}
            {errors.submit && (
              <Alert variant="destructive" className="bg-red-50 border-2 border-red-200">
                <AlertCircle className="h-5 w-5 text-red-600" />
                <AlertDescription className="text-red-800 font-medium">
                  {errors.submit}
                </AlertDescription>
              </Alert>
            )}

            {/* Submit Button */}
            <Button
              type="submit"
              disabled={!isFormValid}
              className={`w-full h-12 text-base font-semibold rounded-lg transition-all ${
                isFormValid
                  ? 'bg-blue-600 hover:bg-blue-700 text-white shadow-lg hover:shadow-xl'
                  : 'bg-gray-300 text-gray-500 cursor-not-allowed'
              }`}
            >
              {uploadMutation.isPending ? (
                <div className="flex items-center justify-center gap-2">
                  <span className="inline-block w-5 h-5 border-3 border-white border-t-transparent rounded-full animate-spin" />
                  <span>Processing...</span>
                </div>
              ) : (
                <span>Upload & Analyze Document</span>
              )}
            </Button>

            {/* Security Notice */}
            <div className="bg-blue-50 rounded-lg p-4 flex gap-3 items-start">
              <Lock className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
              <p className="text-xs text-blue-800">
                <span className="font-semibold">Enterprise Security:</span> Documents are encrypted
                and stored securely. Only accessible to authorized users for case analysis.
              </p>
            </div>
          </form>
        </div>

        {/* Footer */}
        <div className="text-center mt-8 text-sm text-gray-600">
          <p>
            Need help? Visit our <a href="#" className="text-blue-600 hover:underline font-medium">documentation</a>
          </p>
        </div>
      </div>
    </div>
  )
}
