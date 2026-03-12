import api from "./api"
import type { ChunkResponse } from "./types"

// ============================================
// Types matching backend response shapes
// ============================================

export interface MatterResponse {
  id: string
  name: string
  status: "processing" | "ready" | "error" | "cancelled"
  file_type: string
  created_at: string
  updated_at: string | null
}

export interface MatterDetailResponse extends MatterResponse {
  blob_storage_path: string
  documents_count: number
  queries_count: number
}

export interface CreateMatterResponse {
  id: string
  name: string
  status: string
  file_type: string
  documents_count: number
  task_ids: string[]
  created_at: string
}

export interface AskResponse {
  answer: string | null
  sources: {
    chunk_id: string
    page_num: string
    section_name: string
    relevance_score: number
    content: string
    document_id: string
    document_name: string
  }[]
  citations: {
    location: string
    citation_type: string
    relevance_score: number
    chunk_id: string
    supporting_excerpt: string
    is_grounded: boolean
  }[]
  matter_id: string
  query: string
  model: string
  tokens_used: number
  confidence: {
    level: string
    score: number
    factors: {
      has_hallucinations: boolean
      unsupported_claims: number
      grounded_citations: number
      avg_citation_relevance: number
    }
  }
  error: string | null
}

export interface MatterStatusResponse {
  id: string
  name: string
  status: string
  created_at: string
}

// ============================================
// API Service Functions
// ============================================

export async function listMatters(): Promise<MatterResponse[]> {
  const { data } = await api.get<MatterResponse[]>("/matters")
  return data
}

export async function getMatter(id: string): Promise<MatterDetailResponse> {
  const { data } = await api.get<MatterDetailResponse>(`/matters/${id}`)
  return data
}

export async function createMatter(
  name: string,
  files: File[],
  signal?: AbortSignal
): Promise<CreateMatterResponse> {
  const formData = new FormData()
  formData.append("name", name)
  for (const file of files) {
    formData.append("files", file)
  }

  const { data } = await api.post<CreateMatterResponse>("/matters", formData, {
    headers: { "Content-Type": "multipart/form-data" },
    signal,
  })
  return data
}

export async function deleteMatter(id: string): Promise<{ id: string; deleted: boolean }> {
  const { data } = await api.delete(`/matters/${id}`)
  return data
}

export async function cancelMatterProcessing(matterId: string): Promise<{ id: string; cancelled: boolean; status: string }> {
  const { data } = await api.post<{ id: string; cancelled: boolean; status: string }>(
    `/matters/${matterId}/cancel`
  )
  return data
}

export async function askQuestion(matterId: string, question: string): Promise<AskResponse> {
  const { data } = await api.post<AskResponse>(`/matters/${matterId}/ask`, { question })
  return data
}

export async function getMatterStatus(id: string): Promise<MatterStatusResponse> {
  const { data } = await api.get<MatterStatusResponse>(`/matters/${id}/status`)
  return data
}

// ============================================
// Document API Functions
// ============================================

export interface DocumentResponse {
  id: string
  name: string
  file_type: string
  status: string
  chunk_count: number
  summary: string | null
  document_type: string | null
  jurisdiction: string | null
  created_at: string
}

export interface UploadDocumentResponse {
  id: string
  matter_id: string
  name: string
  file_type: string
  status: string
  task_id: string
  created_at: string
}

export async function listMatterDocuments(matterId: string): Promise<DocumentResponse[]> {
  const { data } = await api.get<DocumentResponse[]>(`/matters/${matterId}/documents`)
  return data
}

export async function uploadMatterDocument(
  matterId: string,
  file: File,
  signal?: AbortSignal
): Promise<UploadDocumentResponse> {
  const formData = new FormData()
  formData.append("file", file)

  const { data } = await api.post<UploadDocumentResponse>(
    `/matters/${matterId}/documents`,
    formData,
    { headers: { "Content-Type": "multipart/form-data" }, signal }
  )
  return data
}

export async function fetchMatterChunks(
  matterId: string,
  documentId?: string
): Promise<ChunkResponse[]> {
  const params = documentId ? { document_id: documentId } : {}
  const { data } = await api.get<ChunkResponse[]>(`/matters/${matterId}/chunks`, { params })
  return data
}

export function getMatterDocumentDownloadUrl(matterId: string, documentId: string): string {
  const baseUrl = api.defaults.baseURL || ""
  return `${baseUrl}/matters/${matterId}/documents/${documentId}/download`
}

// ============================================
// Query History API Functions
// ============================================

export interface QueryHistoryItem {
  id: string
  question: string
  answer: string
  citations: AskResponse["sources"] | null
  created_at: string
}

export async function getQueryHistory(
  matterId: string,
  limit = 50
): Promise<QueryHistoryItem[]> {
  const { data } = await api.get<QueryHistoryItem[]>(
    `/matters/${matterId}/queries`,
    { params: { limit } }
  )
  return data
}

// ============================================
// Document Delete API Function
// ============================================

export async function deleteDocument(
  matterId: string,
  documentId: string
): Promise<{ id: string; deleted: boolean }> {
  const { data } = await api.delete(`/matters/${matterId}/documents/${documentId}`)
  return data
}

// ============================================
// SSE Progress Subscription
// ============================================

export function subscribeMatterProgress(matterId: string): EventSource {
  const baseURL = api.defaults.baseURL || ""
  return new EventSource(`${baseURL}/matters/${matterId}/progress`)
}
