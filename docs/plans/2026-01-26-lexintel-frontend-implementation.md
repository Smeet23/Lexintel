# LexIntel Frontend Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a professional legal frontend with shadcn/ui components, React Query data management, and two main pages: Dashboard (upload PDFs) and Case Detail (chat/query interface with citations).

**Architecture:** Next.js 14 with App Router, shadcn/ui components, React Query for async state management, Axios for API calls, TypeScript for type safety, and a lightweight Auth Context for token management. No authentication UI needed (focus on upload/query flows only).

**Tech Stack:**
- Next.js 14 (React 18, TypeScript)
- shadcn/ui (Button, Input, Textarea, Badge, Card, Spinner)
- React Query (@tanstack/react-query)
- Axios for HTTP
- Tailwind CSS + neutral color scheme

**Backend API Contract:**
- `POST /auth/register` - Register user
- `POST /auth/login` - Login, returns `{access_token, token_type}`
- `POST /cases` - Upload PDF (FormData: name, file)
- `GET /cases/{id}/status` - Check processing status
- `POST /cases/{id}/ask` - Query case with question
- No auth required for MVP (demo user hardcoded in backend)

---

## Phase 1: Foundation (API & Auth Context)

### Task 1: Create API client with Axios

**Files:**
- Create: `lib/api.ts`

**Step 1: Write the minimal API client**

Create `lib/api.ts`:

```typescript
import axios from 'axios'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Add auth token to requests
api.interceptors.request.use((config) => {
  const token = typeof window !== 'undefined' ? localStorage.getItem('token') : null
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// Handle 401 responses
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      if (typeof window !== 'undefined') {
        localStorage.removeItem('token')
        window.location.href = '/auth/login'
      }
    }
    return Promise.reject(error)
  }
)

export default api
```

**Step 2: Verify the file exists and has correct TypeScript**

```bash
cat lib/api.ts | head -20
```

**Step 3: Commit**

```bash
git add lib/api.ts
git commit -m "feat: add axios API client with auth interceptor"
```

---

### Task 2: Create types for API responses

**Files:**
- Create: `lib/types.ts`

**Step 1: Write API response types**

Create `lib/types.ts`:

```typescript
export interface User {
  id: string
  email: string
  created_at: string
}

export interface TokenResponse {
  access_token: string
  token_type: string
}

export interface Case {
  id: string
  name: string
  status: 'processing' | 'ready' | 'error'
  created_at: string
  blob_storage_path?: string
}

export interface Citation {
  page: string
  content_snippet: string
  score?: number
}

export interface QueryResponse {
  answer: string
  sources: Citation[]
}

export interface UploadResponse {
  id: string
  name: string
  status: string
  created_at: string
  task_id?: string
}
```

**Step 2: Verify file exists**

```bash
cat lib/types.ts | head -10
```

**Step 3: Commit**

```bash
git add lib/types.ts
git commit -m "feat: add TypeScript types for API responses"
```

---

### Task 3: Create Auth Context for token management

**Files:**
- Create: `lib/auth-context.tsx`

**Step 1: Write Auth Context**

Create `lib/auth-context.tsx`:

```typescript
'use client'

import { createContext, useContext, useState, useEffect, ReactNode } from 'react'

interface AuthContextType {
  token: string | null
  isAuthenticated: boolean
  setToken: (token: string | null) => void
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [token, setTokenState] = useState<string | null>(null)
  const [mounted, setMounted] = useState(false)

  // Load token from localStorage on mount
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const stored = localStorage.getItem('token')
      setTokenState(stored)
    }
    setMounted(true)
  }, [])

  const setToken = (newToken: string | null) => {
    setTokenState(newToken)
    if (typeof window !== 'undefined') {
      if (newToken) {
        localStorage.setItem('token', newToken)
      } else {
        localStorage.removeItem('token')
      }
    }
  }

  // Don't render children until mounted to avoid hydration mismatch
  if (!mounted) {
    return null
  }

  return (
    <AuthContext.Provider value={{ token, isAuthenticated: !!token, setToken }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider')
  }
  return context
}
```

**Step 2: Verify syntax**

```bash
npx tsc --noEmit lib/auth-context.tsx
```

Expected: No errors

**Step 3: Commit**

```bash
git add lib/auth-context.tsx
git commit -m "feat: add auth context for token management"
```

---

## Phase 2: Reusable shadcn/ui Components

### Task 4: Add shadcn/ui Button component

**Files:**
- Create: `components/ui/button.tsx`

**Step 1: Install shadcn Button**

```bash
npx shadcn@latest add button -y
```

**Step 2: Verify component installed**

```bash
ls -la components/ui/button.tsx
```

**Step 3: Commit**

```bash
git add components/ui/button.tsx
git commit -m "feat: add shadcn button component"
```

---

### Task 5: Add shadcn/ui Input component

**Files:**
- Create: `components/ui/input.tsx`

**Step 1: Install shadcn Input**

```bash
npx shadcn@latest add input -y
```

**Step 2: Verify component installed**

```bash
ls -la components/ui/input.tsx
```

**Step 3: Commit**

```bash
git add components/ui/input.tsx
git commit -m "feat: add shadcn input component"
```

---

### Task 6: Add shadcn/ui Textarea component

**Files:**
- Create: `components/ui/textarea.tsx`

**Step 1: Install shadcn Textarea**

```bash
npx shadcn@latest add textarea -y
```

**Step 2: Verify component installed**

```bash
ls -la components/ui/textarea.tsx
```

**Step 3: Commit**

```bash
git add components/ui/textarea.tsx
git commit -m "feat: add shadcn textarea component"
```

---

### Task 7: Add shadcn/ui Badge component

**Files:**
- Create: `components/ui/badge.tsx`

**Step 1: Install shadcn Badge**

```bash
npx shadcn@latest add badge -y
```

**Step 2: Verify component installed**

```bash
ls -la components/ui/badge.tsx
```

**Step 3: Commit**

```bash
git add components/ui/badge.tsx
git commit -m "feat: add shadcn badge component"
```

---

### Task 8: Add shadcn/ui Card component

**Files:**
- Create: `components/ui/card.tsx`

**Step 1: Install shadcn Card**

```bash
npx shadcn@latest add card -y
```

**Step 2: Verify component installed**

```bash
ls -la components/ui/card.tsx
```

**Step 3: Commit**

```bash
git add components/ui/card.tsx
git commit -m "feat: add shadcn card component"
```

---

### Task 9: Create Spinner loading component

**Files:**
- Create: `components/spinner.tsx`

**Step 1: Write Spinner component**

Create `components/spinner.tsx`:

```typescript
'use client'

export function Spinner() {
  return (
    <div className="inline-block">
      <div className="relative h-5 w-5">
        <div className="absolute inset-0 rounded-full border-2 border-slate-200 border-t-slate-900 animate-spin" />
      </div>
    </div>
  )
}

export function SpinnerWithText({ text }: { text?: string }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-8">
      <Spinner />
      {text && <p className="text-sm text-slate-600">{text}</p>}
    </div>
  )
}
```

**Step 2: Verify component**

```bash
cat components/spinner.tsx
```

**Step 3: Commit**

```bash
git add components/spinner.tsx
git commit -m "feat: add spinner loading component"
```

---

### Task 10: Create FileUploadZone component

**Files:**
- Create: `components/file-upload-zone.tsx`

**Step 1: Write FileUploadZone component**

Create `components/file-upload-zone.tsx`:

```typescript
'use client'

import { useState, useRef } from 'react'
import { Upload, AlertCircle } from 'lucide-react'
import { Alert, AlertDescription } from './ui/alert'

interface FileUploadZoneProps {
  onFileSelect: (file: File) => void
  isLoading?: boolean
}

export function FileUploadZone({ onFileSelect, isLoading }: FileUploadZoneProps) {
  const [isDragActive, setIsDragActive] = useState(false)
  const [error, setError] = useState<string>('')
  const inputRef = useRef<HTMLInputElement>(null)

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (!isLoading) {
      setIsDragActive(e.type === 'dragenter' || e.type === 'dragover')
    }
  }

  const handleValidateAndSelect = (file: File) => {
    setError('')

    // Validate PDF
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setError('Only PDF files are allowed')
      return
    }

    // Validate size (max 50MB)
    const maxSize = 50 * 1024 * 1024
    if (file.size > maxSize) {
      setError('File size must be less than 50MB')
      return
    }

    onFileSelect(file)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragActive(false)

    if (isLoading) return

    const file = e.dataTransfer.files?.[0]
    if (file) {
      handleValidateAndSelect(file)
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.currentTarget.files?.[0]
    if (file) {
      handleValidateAndSelect(file)
    }
  }

  return (
    <div
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
      onClick={() => !isLoading && inputRef.current?.click()}
      className={`relative rounded-lg border-2 border-dashed p-8 text-center cursor-pointer transition-colors ${
        isDragActive
          ? 'border-slate-900 bg-slate-50'
          : 'border-slate-300 hover:border-slate-400 hover:bg-slate-50'
      } ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".pdf"
        onChange={handleFileChange}
        disabled={isLoading}
        className="hidden"
      />

      <Upload className="mx-auto h-10 w-10 text-slate-600 mb-3" />
      <p className="text-slate-900 font-medium">Drag case PDF here or click to browse</p>
      <p className="text-sm text-slate-500 mt-1">PDF only, max 50MB</p>

      {error && (
        <Alert variant="destructive" className="mt-4">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}
    </div>
  )
}
```

**Step 2: Install Alert component (needed for error display)**

```bash
npx shadcn@latest add alert -y
```

**Step 3: Verify component**

```bash
cat components/file-upload-zone.tsx | head -20
```

**Step 4: Commit**

```bash
git add components/file-upload-zone.tsx components/ui/alert.tsx
git commit -m "feat: add file upload zone and alert components"
```

---

## Phase 3: Main Pages

### Task 11: Update root layout with AuthProvider

**Files:**
- Modify: `app/layout.tsx`

**Step 1: Update layout with AuthProvider and navbar**

Read current layout first to understand structure, then update:

```bash
cat app/layout.tsx
```

Replace `app/layout.tsx` content with:

```typescript
import type { Metadata } from 'next'
import { AuthProvider } from '@/lib/auth-context'
import './globals.css'

export const metadata: Metadata = {
  title: 'LexIntel - Legal RAG',
  description: 'RAG system for legal document analysis',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="bg-slate-50">
        <AuthProvider>
          <nav className="bg-white shadow sticky top-0 z-50">
            <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
              <a href="/" className="text-2xl font-bold text-slate-900 flex items-center gap-2">
                <div className="bg-slate-900 text-white rounded px-2 py-1 text-sm font-mono">LI</div>
                LexIntel
              </a>
              <div className="flex items-center gap-4">
                <a href="/dashboard" className="text-slate-700 hover:text-slate-900 font-medium">
                  Dashboard
                </a>
                <button
                  onClick={() => {
                    localStorage.removeItem('token')
                    window.location.href = '/'
                  }}
                  className="text-slate-700 hover:text-slate-900 font-medium"
                >
                  Logout
                </button>
              </div>
            </div>
          </nav>
          <main className="max-w-7xl mx-auto px-4 py-8">
            {children}
          </main>
        </AuthProvider>
      </body>
    </html>
  )
}
```

**Step 2: Verify TypeScript**

```bash
npx tsc --noEmit app/layout.tsx
```

Expected: No errors

**Step 3: Commit**

```bash
git add app/layout.tsx
git commit -m "feat: wrap app with auth provider and update navbar"
```

---

### Task 12: Create Dashboard page (upload PDF)

**Files:**
- Create: `app/dashboard/page.tsx`

**Step 1: Write Dashboard page**

Create `app/dashboard/page.tsx`:

```typescript
'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { useMutation } from '@tanstack/react-query'
import api from '@/lib/api'
import { FileUploadZone } from '@/components/file-upload-zone'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Spinner } from '@/components/spinner'
import { AlertCircle } from 'lucide-react'

export default function Dashboard() {
  const router = useRouter()
  const [caseName, setCaseName] = useState('')
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [error, setError] = useState('')

  const uploadMutation = useMutation({
    mutationFn: async () => {
      if (!selectedFile || !caseName.trim()) {
        throw new Error('Please select a file and enter a case name')
      }

      const formData = new FormData()
      formData.append('name', caseName)
      formData.append('file', selectedFile)

      const response = await api.post('/cases', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      return response.data
    },
    onSuccess: (data) => {
      // Redirect to case detail page
      router.push(`/cases/${data.id}`)
    },
    onError: (err: any) => {
      setError(err.response?.data?.detail || 'Failed to upload case. Please try again.')
    },
  })

  const handleUpload = () => {
    setError('')
    uploadMutation.mutate()
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-slate-900">Dashboard</h1>
        <p className="text-slate-600 mt-2">Upload legal documents for analysis</p>
      </div>

      <Card className="border-slate-200">
        <CardHeader>
          <CardTitle>Upload New Case</CardTitle>
          <CardDescription>Upload a PDF document to analyze</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-slate-900 mb-2">
              Case Name
            </label>
            <Input
              placeholder="e.g., Smith v. Insurance Co."
              value={caseName}
              onChange={(e) => setCaseName(e.target.value)}
              disabled={uploadMutation.isPending}
              className="border-slate-300"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-900 mb-2">
              Case Document
            </label>
            <FileUploadZone
              onFileSelect={setSelectedFile}
              isLoading={uploadMutation.isPending}
            />
            {selectedFile && (
              <p className="text-sm text-slate-600 mt-2">
                Selected: {selectedFile.name} ({(selectedFile.size / 1024 / 1024).toFixed(2)} MB)
              </p>
            )}
          </div>

          {error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          <Button
            onClick={handleUpload}
            disabled={!caseName.trim() || !selectedFile || uploadMutation.isPending}
            className="w-full bg-slate-900 hover:bg-slate-800"
            size="lg"
          >
            {uploadMutation.isPending ? (
              <>
                <Spinner /> Uploading...
              </>
            ) : (
              'Upload & Analyze'
            )}
          </Button>
        </CardContent>
      </Card>
    </div>
  )
}
```

**Step 2: Verify TypeScript**

```bash
npx tsc --noEmit app/dashboard/page.tsx
```

Expected: No errors

**Step 3: Build to check for runtime errors**

```bash
npm run build 2>&1 | grep -A 5 "error\|Error" || echo "Build successful"
```

**Step 4: Commit**

```bash
git add app/dashboard/page.tsx
git commit -m "feat: create dashboard with PDF upload form"
```

---

### Task 13: Create Case Detail page (query interface)

**Files:**
- Create: `app/cases/[id]/page.tsx`

**Step 1: Write Case Detail page**

Create `app/cases/[id]/page.tsx`:

```typescript
'use client'

import { useState, useEffect } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import api from '@/lib/api'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { SpinnerWithText, Spinner } from '@/components/spinner'
import { AlertCircle, Copy, Check } from 'lucide-react'
import { type Case, type QueryResponse, type Citation } from '@/lib/types'

export default function CaseDetail({ params }: { params: { id: string } }) {
  const { id: caseId } = params
  const [question, setQuestion] = useState('')
  const [copied, setCopied] = useState(false)

  // Poll case status
  const {
    data: caseData,
    isLoading: caseLoading,
    error: caseError,
    refetch: refetchCase,
  } = useQuery({
    queryKey: ['case', caseId],
    queryFn: async () => {
      const response = await api.get(`/cases/${caseId}/status`)
      return response.data as Case
    },
    refetchInterval: (data) => {
      // Stop polling when status is not "processing"
      return data?.status === 'processing' ? 2000 : false
    },
    refetchIntervalInBackground: false,
  })

  // Ask question mutation
  const queryMutation = useMutation({
    mutationFn: async (q: string) => {
      const response = await api.post(`/cases/${caseId}/ask`, {
        question: q,
      })
      return response.data as QueryResponse
    },
  })

  const handleAsk = () => {
    if (!question.trim()) return
    queryMutation.mutate(question)
  }

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  if (caseLoading) {
    return <SpinnerWithText text="Loading case..." />
  }

  if (caseError) {
    return (
      <Alert variant="destructive">
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>Failed to load case. Please try again.</AlertDescription>
      </Alert>
    )
  }

  if (!caseData) {
    return (
      <Alert variant="destructive">
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>Case not found.</AlertDescription>
      </Alert>
    )
  }

  const isProcessing = caseData.status === 'processing'
  const isError = caseData.status === 'error'
  const isReady = caseData.status === 'ready'

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">{caseData.name}</h1>
            <p className="text-slate-600 mt-2">
              Uploaded {new Date(caseData.created_at).toLocaleDateString()}
            </p>
          </div>
          <Badge
            variant={
              isProcessing ? 'secondary' : isError ? 'destructive' : 'default'
            }
            className={
              isReady
                ? 'bg-green-600 hover:bg-green-700 text-white'
                : isError
                ? ''
                : ''
            }
          >
            {isProcessing && (
              <>
                <Spinner /> Processing...
              </>
            )}
            {isError && 'Error'}
            {isReady && 'Ready'}
          </Badge>
        </div>
      </div>

      {/* Status messages */}
      {isProcessing && (
        <Alert className="border-blue-200 bg-blue-50">
          <AlertDescription className="text-blue-900">
            Document is being analyzed. Please wait...
          </AlertDescription>
        </Alert>
      )}

      {isError && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            Document processing failed. Please re-upload the document.
          </AlertDescription>
        </Alert>
      )}

      {/* Query Interface (only show when ready) */}
      {isReady && (
        <div className="grid gap-8 lg:grid-cols-3">
          {/* Query Input */}
          <div className="lg:col-span-1">
            <Card className="border-slate-200">
              <CardHeader>
                <CardTitle>Ask a Question</CardTitle>
                <CardDescription>Query the document</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <Textarea
                  placeholder="Ask a legal question about the case..."
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  disabled={queryMutation.isPending}
                  className="min-h-24 border-slate-300"
                />
                <Button
                  onClick={handleAsk}
                  disabled={!question.trim() || queryMutation.isPending}
                  className="w-full bg-slate-900 hover:bg-slate-800"
                >
                  {queryMutation.isPending ? (
                    <>
                      <Spinner /> Analyzing...
                    </>
                  ) : (
                    'Ask LexIntel'
                  )}
                </Button>
              </CardContent>
            </Card>
          </div>

          {/* Query Results */}
          <div className="lg:col-span-2">
            {queryMutation.isPending && (
              <SpinnerWithText text="Analyzing document..." />
            )}

            {queryMutation.isSuccess && queryMutation.data && (
              <Card className="border-slate-200">
                <CardHeader>
                  <CardTitle>Answer</CardTitle>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="bg-slate-50 rounded-lg p-4 border border-slate-200">
                    <p className="text-slate-900 leading-relaxed whitespace-pre-wrap">
                      {queryMutation.data.answer}
                    </p>
                  </div>

                  {queryMutation.data.sources && queryMutation.data.sources.length > 0 && (
                    <div>
                      <h3 className="font-semibold text-slate-900 mb-3">
                        Sources & Citations
                      </h3>
                      <div className="space-y-2">
                        {queryMutation.data.sources.map((citation: Citation, idx: number) => (
                          <div
                            key={idx}
                            className="bg-slate-50 rounded-lg p-3 border border-slate-200 hover:border-slate-300 transition-colors"
                          >
                            <div className="flex items-start justify-between gap-2 mb-2">
                              <Badge variant="secondary" className="bg-blue-100 text-blue-900">
                                Page {citation.page}
                              </Badge>
                              <button
                                onClick={() => copyToClipboard(citation.content_snippet)}
                                className="p-1 hover:bg-slate-200 rounded"
                                title="Copy citation"
                              >
                                {copied ? (
                                  <Check className="h-4 w-4 text-green-600" />
                                ) : (
                                  <Copy className="h-4 w-4 text-slate-500" />
                                )}
                              </button>
                            </div>
                            <p className="text-sm text-slate-700 line-clamp-3">
                              {citation.content_snippet}
                            </p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            )}

            {queryMutation.isError && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>
                  {queryMutation.error instanceof Error
                    ? queryMutation.error.message
                    : 'Failed to process query. Please try again.'}
                </AlertDescription>
              </Alert>
            )}

            {!queryMutation.isPending &&
              !queryMutation.isSuccess &&
              !queryMutation.isError && (
                <Card className="border-slate-200 border-dashed">
                  <CardContent className="text-center py-12 text-slate-600">
                    <p>Ask a question to get started</p>
                  </CardContent>
                </Card>
              )}
          </div>
        </div>
      )}
    </div>
  )
}
```

**Step 2: Verify TypeScript**

```bash
npx tsc --noEmit app/cases/[id]/page.tsx
```

Expected: No errors

**Step 3: Build**

```bash
npm run build 2>&1 | grep -A 5 "error\|Error" || echo "Build successful"
```

**Step 4: Commit**

```bash
git add app/cases/[id]/page.tsx
git commit -m "feat: create case detail page with query interface"
```

---

### Task 14: Update home page with login redirect

**Files:**
- Modify: `app/page.tsx`

**Step 1: Update home page to redirect to dashboard if authenticated**

Replace `app/page.tsx` content:

```typescript
'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { useAuth } from '@/lib/auth-context'
import Link from 'next/link'

export default function Home() {
  const router = useRouter()
  const { isAuthenticated } = useAuth()

  useEffect(() => {
    if (isAuthenticated) {
      router.push('/dashboard')
    }
  }, [isAuthenticated, router])

  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] text-center space-y-8">
      <div>
        <h1 className="text-5xl font-bold text-slate-900 mb-4">LexIntel</h1>
        <p className="text-xl text-slate-600">Legal Document Analysis with RAG</p>
      </div>

      <p className="text-slate-600 max-w-md">
        Upload legal documents and ask intelligent questions powered by retrieval-augmented generation.
      </p>

      <div className="flex gap-4">
        <Link
          href="/auth/login"
          className="px-8 py-3 bg-slate-900 text-white rounded-lg font-medium hover:bg-slate-800 transition-colors"
        >
          Login
        </Link>
        <Link
          href="/auth/register"
          className="px-8 py-3 bg-slate-200 text-slate-900 rounded-lg font-medium hover:bg-slate-300 transition-colors"
        >
          Register
        </Link>
      </div>
    </div>
  )
}
```

**Step 2: Verify TypeScript**

```bash
npx tsc --noEmit app/page.tsx
```

**Step 3: Build**

```bash
npm run build 2>&1 | tail -10
```

**Step 4: Commit**

```bash
git add app/page.tsx
git commit -m "feat: update home page with auth redirect"
```

---

### Task 15: Create Auth Login page

**Files:**
- Create: `app/auth/login/page.tsx`

**Step 1: Write login page**

Create `app/auth/login/page.tsx`:

```typescript
'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { useMutation } from '@tanstack/react-query'
import api from '@/lib/api'
import { useAuth } from '@/lib/auth-context'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { AlertCircle } from 'lucide-react'
import Link from 'next/link'

export default function Login() {
  const router = useRouter()
  const { setToken } = useAuth()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')

  const loginMutation = useMutation({
    mutationFn: async () => {
      const response = await api.post('/auth/login', { email, password })
      return response.data
    },
    onSuccess: (data) => {
      setToken(data.access_token)
      router.push('/dashboard')
    },
    onError: (err: any) => {
      setError(err.response?.data?.detail || 'Login failed. Please try again.')
    },
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    loginMutation.mutate()
  }

  return (
    <div className="flex items-center justify-center min-h-[60vh]">
      <Card className="w-full max-w-md border-slate-200">
        <CardHeader>
          <CardTitle>Login</CardTitle>
          <CardDescription>Sign in to your LexIntel account</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-slate-900 mb-2">
                Email
              </label>
              <Input
                type="email"
                placeholder="you@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                disabled={loginMutation.isPending}
                className="border-slate-300"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-900 mb-2">
                Password
              </label>
              <Input
                type="password"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                disabled={loginMutation.isPending}
                className="border-slate-300"
              />
            </div>

            {error && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            <Button
              type="submit"
              disabled={!email || !password || loginMutation.isPending}
              className="w-full bg-slate-900 hover:bg-slate-800"
            >
              {loginMutation.isPending ? 'Signing in...' : 'Sign In'}
            </Button>
          </form>

          <p className="text-sm text-slate-600 text-center mt-4">
            Don't have an account?{' '}
            <Link href="/auth/register" className="text-slate-900 hover:underline font-medium">
              Register
            </Link>
          </p>
        </CardContent>
      </Card>
    </div>
  )
}
```

**Step 2: Verify TypeScript**

```bash
npx tsc --noEmit app/auth/login/page.tsx
```

**Step 3: Build**

```bash
npm run build 2>&1 | tail -5
```

**Step 4: Commit**

```bash
git add app/auth/login/page.tsx
git commit -m "feat: create login page with form and auth flow"
```

---

### Task 16: Create Auth Register page

**Files:**
- Create: `app/auth/register/page.tsx`

**Step 1: Write register page**

Create `app/auth/register/page.tsx`:

```typescript
'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { useMutation } from '@tanstack/react-query'
import api from '@/lib/api'
import { useAuth } from '@/lib/auth-context'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { AlertCircle } from 'lucide-react'
import Link from 'next/link'

export default function Register() {
  const router = useRouter()
  const { setToken } = useAuth()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [error, setError] = useState('')

  const registerMutation = useMutation({
    mutationFn: async () => {
      if (password !== confirmPassword) {
        throw new Error('Passwords do not match')
      }
      const response = await api.post('/auth/register', { email, password })
      return response.data
    },
    onSuccess: () => {
      // Auto-login after registration
      loginMutation.mutate()
    },
    onError: (err: any) => {
      setError(err.response?.data?.detail || err.message || 'Registration failed')
    },
  })

  const loginMutation = useMutation({
    mutationFn: async () => {
      const response = await api.post('/auth/login', { email, password })
      return response.data
    },
    onSuccess: (data) => {
      setToken(data.access_token)
      router.push('/dashboard')
    },
    onError: (err: any) => {
      setError('Registration successful but login failed. Please try logging in.')
    },
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    setError('')

    if (password !== confirmPassword) {
      setError('Passwords do not match')
      return
    }

    if (password.length < 6) {
      setError('Password must be at least 6 characters')
      return
    }

    registerMutation.mutate()
  }

  const isLoading = registerMutation.isPending || loginMutation.isPending

  return (
    <div className="flex items-center justify-center min-h-[60vh]">
      <Card className="w-full max-w-md border-slate-200">
        <CardHeader>
          <CardTitle>Create Account</CardTitle>
          <CardDescription>Sign up for LexIntel</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-slate-900 mb-2">
                Email
              </label>
              <Input
                type="email"
                placeholder="you@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                disabled={isLoading}
                className="border-slate-300"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-900 mb-2">
                Password
              </label>
              <Input
                type="password"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                disabled={isLoading}
                className="border-slate-300"
              />
              <p className="text-xs text-slate-500 mt-1">Minimum 6 characters</p>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-900 mb-2">
                Confirm Password
              </label>
              <Input
                type="password"
                placeholder="••••••••"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                disabled={isLoading}
                className="border-slate-300"
              />
            </div>

            {error && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            <Button
              type="submit"
              disabled={!email || !password || !confirmPassword || isLoading}
              className="w-full bg-slate-900 hover:bg-slate-800"
            >
              {isLoading ? 'Creating account...' : 'Create Account'}
            </Button>
          </form>

          <p className="text-sm text-slate-600 text-center mt-4">
            Already have an account?{' '}
            <Link href="/auth/login" className="text-slate-900 hover:underline font-medium">
              Login
            </Link>
          </p>
        </CardContent>
      </Card>
    </div>
  )
}
```

**Step 2: Verify TypeScript**

```bash
npx tsc --noEmit app/auth/register/page.tsx
```

**Step 3: Build**

```bash
npm run build 2>&1 | tail -5
```

**Step 4: Commit**

```bash
git add app/auth/register/page.tsx
git commit -m "feat: create register page with validation"
```

---

### Task 17: Create .env.local for API configuration

**Files:**
- Create: `.env.local`

**Step 1: Create environment file**

Create `frontend/.env.local`:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

**Step 2: Verify file exists**

```bash
cat .env.local
```

**Step 3: Commit**

```bash
git add .env.local
git commit -m "chore: add environment configuration"
```

---

### Task 18: Final build and test

**Files:**
- No new files

**Step 1: Run full build**

```bash
npm run build
```

Expected: Successfully compiled

**Step 2: Check for TypeScript errors**

```bash
npx tsc --noEmit
```

Expected: No errors

**Step 3: Final commit check**

```bash
git status
```

Expected: Nothing to commit

---

## End State

After completing all 18 tasks:

✅ **API Layer:**
- Axios client with auth interceptor
- Auth context for token management
- Full TypeScript types for API responses

✅ **UI Components:**
- shadcn/ui: Button, Input, Textarea, Badge, Card, Alert
- Custom: FileUploadZone, Spinner

✅ **Pages:**
- `/` - Home with login/register links
- `/auth/login` - Login form with auth flow
- `/auth/register` - Registration with validation
- `/dashboard` - PDF upload interface
- `/cases/[id]` - Chat/query interface with status polling

✅ **Features:**
- PDF upload with validation and progress
- Real-time status polling while processing
- Query interface with React Query
- Citation display with copy functionality
- Professional design with neutral color scheme
- Full TypeScript type safety

---

## Testing Workflow (Manual)

1. **Start backend:** `uvicorn backend.main:app --reload`
2. **Start frontend:** `npm run dev`
3. **Test flow:**
   - Visit http://localhost:3000
   - Register new account
   - Upload PDF on dashboard
   - Watch status polling
   - Ask questions about case
   - Verify citations display

