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
