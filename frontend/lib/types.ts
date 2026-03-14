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

// ============================================
// Contract Review Types
// ============================================

export interface ContractRisk {
  clause: string
  risk_level: "high" | "medium" | "low"
  explanation: string
  remedy: string
}

export interface ContractReviewResult {
  exists: boolean
  id?: string
  matter_id?: string
  document_id?: string
  risks?: ContractRisk[]
  summary?: {
    total_clauses: number
    high_risk: number
    medium_risk: number
    low_risk: number
  }
  missing_clauses?: string[]
  overall_score?: number
  created_at?: string
}

// ============================================
// Draft Types
// ============================================

export interface DraftSource {
  document_name: string
  page_num: string
  section_name: string
  excerpt: string
}

export interface DraftResponse {
  id: string
  document_type: string
  instructions: string
  content: string
  sources: DraftSource[]
  created_at: string
}

// ============================================
// Audit Log Types
// ============================================

export interface AuditLogEntry {
  id: string
  action: string
  user: string
  details: string | null
  sources: string | null
  created_at: string
}

// ============================================
// Precedent Types
// ============================================

export interface PrecedentSearchResult {
  matter_id: string
  matter_name: string
  document_name: string
  page_num: string
  section_name: string
  content: string
  relevance_score: number
}

export interface SavedPrecedent {
  id: string
  title: string
  query: string
  document_name: string | null
  matter_id: string | null
  chunk_content: string | null
  page_num: number | null
  section_name: string | null
  relevance_score: string | null
  tags: string[]
  notes: string | null
  created_at: string
}
