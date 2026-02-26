export interface User {
  id: string
  name: string
  email: string
  role: "partner" | "associate" | "paralegal" | "admin"
  avatar?: string
}

export interface Matter {
  id: string
  title: string
  jurisdiction: string
  status: "active" | "review" | "closed" | "archived"
  team: string[]
  documentsCount: number
  queriesCount: number
  tokenUsage: number
  budget: number
  lastActivity: string
  createdAt: string
}

export interface Document {
  id: string
  matterId: string
  name: string
  fileType: "pdf" | "docx" | "txt"
  status: "uploading" | "processing" | "indexed" | "error"
  pageCount?: number
  size: number
  uploadedAt: string
  uploadedBy: string
}

export interface Citation {
  documentName: string
  pageNumber: number
  section?: string
  excerpt: string
  relevanceScore: number
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

export interface Precedent {
  id: string
  title: string
  jurisdiction: string
  practiceArea: string
  matterId: string
  matterTitle: string
  content: string
  tags: string[]
  savedBy: string
  savedAt: string
}

export interface TeamMember {
  id: string
  name: string
  email: string
  role: "partner" | "associate" | "paralegal"
  matters: number
  queries: number
  lastActive: string
  status: "active" | "invited" | "deactivated"
}

export interface BillingUsage {
  date: string
  tokens: number
  queries: number
  cost: number
}

export interface Invoice {
  id: string
  date: string
  amount: number
  status: "paid" | "pending" | "overdue"
  downloadUrl: string
}

export interface DashboardStats {
  activeMatters: number
  tokenUsage: number
  pendingReviews: number
  recentQueries: number
}

// ============================================
// Backend API Response Types
// ============================================

export interface CaseResponse {
  id: string
  name: string
  status: "processing" | "ready" | "error"
  file_type: string
  created_at: string
  updated_at: string | null
}

export interface CreateCaseResponse {
  id: string
  name: string
  status: string
  file_type: string
  blob_storage_path: string
  task_id: string
  created_at: string
}

export interface CaseStatusResponse {
  id: string
  name: string
  status: "processing" | "ready" | "error"
  file_type?: string
  created_at: string
}

export interface RAGSource {
  chunk_id: string
  page_num: string
  relevance_score: number
  content: string
}

export interface RAGResponse {
  answer: string | null
  sources: RAGSource[]
  case_id: string
  query: string
  model: string
  tokens_used: number
  confidence: {
    level: string
    score: number
    factors: Record<string, unknown>
  }
  error: string | null
}

export interface ProgressEvent {
  stage: string
  progress: number
  current?: number
  total?: number
  detail?: string
}

export interface ChunkResponse {
  id: string
  page_num: string
  section_name: string
  section_type: string
  content: string
  chunk_sequence: number
}

// ============================================
// Adapter Functions: Backend → Frontend
// ============================================

export function caseToMatter(c: CaseResponse): Matter {
  const statusMap: Record<string, Matter["status"]> = {
    processing: "review",
    ready: "active",
    error: "closed",
  }
  return {
    id: c.id,
    title: c.name,
    jurisdiction: "",
    status: statusMap[c.status] || "active",
    team: [],
    documentsCount: 1,
    queriesCount: 0,
    tokenUsage: 0,
    budget: 0,
    lastActivity: c.updated_at || c.created_at,
    createdAt: c.created_at,
  }
}

export function ragSourceToCitation(source: RAGSource, caseName: string): Citation {
  return {
    documentName: caseName,
    pageNumber: parseInt(source.page_num, 10) || 0,
    excerpt: source.content.slice(0, 200),
    relevanceScore: source.relevance_score,
  }
}

export function ragResponseToMessage(response: RAGResponse, caseName: string): QueryMessage {
  return {
    id: `msg-${Date.now()}`,
    role: "assistant",
    content: response.answer || "I could not generate an answer. Please try again.",
    citations: response.sources.map((s) => ragSourceToCitation(s, caseName)),
    confidenceScore: Math.round(response.confidence.score * 100),
    timestamp: new Date().toISOString(),
  }
}
