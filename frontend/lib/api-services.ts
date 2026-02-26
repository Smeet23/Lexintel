import api from "./api"

// ============================================
// Types matching backend response shapes
// ============================================

export interface MatterResponse {
  id: string
  name: string
  status: "processing" | "ready" | "error"
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
  blob_storage_path: string
  task_id: string
  created_at: string
}

export interface AskResponse {
  answer: string | null
  sources: {
    chunk_id: string
    page_num: string
    relevance_score: number
    content: string
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

export async function createMatter(name: string, file: File): Promise<CreateMatterResponse> {
  const formData = new FormData()
  formData.append("name", name)
  formData.append("file", file)

  const { data } = await api.post<CreateMatterResponse>("/matters", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  })
  return data
}

export async function deleteMatter(id: string): Promise<{ id: string; deleted: boolean }> {
  const { data } = await api.delete(`/matters/${id}`)
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
