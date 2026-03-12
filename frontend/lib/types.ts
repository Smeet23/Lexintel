export interface Citation {
  documentName: string
  pageNumber: number
  section?: string
  excerpt: string
  relevanceScore: number
  /** Full chunk content for click-to-view */
  content?: string
}

export interface QueryMessage {
  id: string
  role: "user" | "assistant"
  content: string
  citations?: Citation[]
  confidenceScore?: number
  timestamp: string
}

export interface AuditEntry {
  id: string
  action: string
  user: string
  details: string
  sources?: string[]
  timestamp: string
}

export interface ProgressEvent {
  stage: string
  step: number
  total_steps: number
  progress: number
  message: string
  current?: number
  total?: number
  detail?: string
  retry_attempt?: number
  error?: string
}

export interface ChunkResponse {
  id: string
  page_num: string
  section_name: string
  section_type: string
  content: string
  concepts: string[]
  chunk_sequence: number
}
