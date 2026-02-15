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
// FIRM & THEME TYPES
// ============================================

export interface ThemeColorTokens {
  [key: string]: string
}

export interface ThemeTypography {
  "font-sans"?: string
  "font-display"?: string
  "font-mono"?: string
}

export interface ThemeLayout {
  "radius-sm"?: string
  "radius-md"?: string
  "radius-lg"?: string
  "radius-xl"?: string
}

export interface ThemeShadows {
  color?: string
  opacity?: string
  blur?: string
  spread?: string
  "offset-x"?: string
  "offset-y"?: string
}

export interface ThemeConfig {
  light: ThemeColorTokens
  dark: ThemeColorTokens
  typography?: ThemeTypography
  layout?: ThemeLayout
  shadows?: ThemeShadows
}

export interface FirmThemeResponse {
  firm_name: string
  firm_slug: string
  logo_url: string | null
  theme: ThemeConfig
}

export interface FirmResponse {
  id: string
  name: string
  slug: string
  logo_url: string | null
  theme_config: ThemeConfig | null
  created_at: string
  updated_at: string
}

export interface FirmMember {
  id: string
  email: string
  name: string | null
  role: "admin" | "partner" | "associate" | "paralegal"
}
