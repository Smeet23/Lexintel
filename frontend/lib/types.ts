import type {
  RawCitationVerification,
  RawClaimVerification,
  RawConflictAnalysis,
} from "./api-services"

// re-export so consumers can import from one place if needed
export type { RawCitationVerification, RawClaimVerification, RawConflictAnalysis }

// ============================================
// Citation Verification Types
// ============================================

export type VerificationStatus = "verified" | "partial" | "unverified" | "not_found" | "unverifiable"
export type CaseStatus = "good_law" | "caution" | "bad_law" | "unknown"

export interface CitationVerification {
  index: number
  rawText: string
  caseName?: string
  court?: string
  date?: string
  jurisdiction?: string
  status: VerificationStatus
  caseStatus?: CaseStatus
  confidence: number
  sourceUrl?: string
  citationCount?: number
  quoteMatch?: number
  matchedSourceIdx?: number
  verificationMethod?: string
}

export interface VerificationSummary {
  total: number
  verified: number
  partial: number
  unverified: number
  not_found: number
  unverifiable: number
}

// ============================================
// Citation & Message Types
// ============================================

export interface Citation {
  documentName: string
  pageNumber: number
  section?: string
  excerpt: string
  relevanceScore: number
  /** Full chunk content for click-to-view */
  content?: string
  /** "document" or "case_law" */
  sourceType?: "document" | "case_law"
  /** CourtListener URL for case law */
  url?: string
  /** Citation verification data (if available) */
  verification?: CitationVerification
  /** Authority hierarchy metadata */
  courtLevel?: string
  jurisdictionCode?: string
  authorityScore?: number
  bindingAuthority?: boolean
  /** Temporal metadata */
  effectiveDate?: string | null
  documentStatus?: string
}

// ============================================
// Claim Verification Types (Grounding)
// ============================================

export type ClaimStatus = "supported" | "partially_supported" | "unsupported" | "uncited"

export interface ClaimVerification {
  claimText: string
  citedSources: number[]
  status: ClaimStatus
  confidence: number
  verificationTier: number
  issues: string[]
  sourceExcerpt?: string
  requiresReview?: boolean
}

export interface ClaimVerificationSummary {
  total: number
  supported: number
  partially_supported: number
  unsupported: number
  uncited: number
}

// ============================================
// Conflict Detection Types
// ============================================

export interface ConflictCredibility {
  score: number
  authority_type: string
  authority_weight: number
  recency_weight: number
  specificity_weight: number
  citation_weight: number
}

export interface ConflictChunkInfo {
  document_id: string
  document_name: string
  page_num: string
  section_name?: string
  content_snippet: string
  credibility: ConflictCredibility
}

export interface ConflictItem {
  id: string
  severity: "high" | "medium"
  contradiction_score: number
  chunk_a: ConflictChunkInfo
  chunk_b: ConflictChunkInfo
  recommended_source: "a" | "b"
  explanation: string
}

export interface ConflictSummary {
  total_conflicts: number
  high_severity: number
  medium_severity: number
  recommended_action: string
}

export interface ConflictAnalysis {
  has_conflicts: boolean
  conflicts: ConflictItem[]
  summary: ConflictSummary
}

// ============================================
// Message Types
// ============================================

export interface QueryMessage {
  id: string
  role: "user" | "assistant"
  content: string
  citations?: Citation[]
  confidenceScore?: number
  timestamp: string
  queryId?: string
  /** Citation verification results */
  verificationSummary?: VerificationSummary
  verifiedCitations?: CitationVerification[]
  /** Claim verification results (grounding) */
  claimVerificationSummary?: ClaimVerificationSummary
  verifiedClaims?: ClaimVerification[]
  /** Conflict detection results */
  conflictAnalysis?: ConflictAnalysis
  /** Issue analysis (problem formulation) */
  issueAnalysis?: IssueAnalysis
}

export interface ProgressEvent {
  stage: string
  step: number
  total_steps: number
  progress: number
  overall_progress: number
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
// Problem Formulation Types (Issue Classification)
// ============================================

export interface IdentifiedIssue {
  domain: string
  issue: string
  legalQuestion: string
  confidence: number
  keyFacts: string[]
}

export interface IssueAnalysis {
  issues: IdentifiedIssue[]
  missingInformation: string[]
}

// ============================================
// Citation Graph Types
// ============================================

export type TreatmentType = "FOLLOWS" | "AFFIRMS" | "APPLIES" | "OVERRULES" | "REVERSES" | "ABROGATES" | "SUPERSEDES" | "QUESTIONS" | "CRITICIZES" | "LIMITS" | "CITES" | "DISTINGUISHES" | "EXPLAINS" | "HARMONIZES" | "AMENDS" | "REPEALS" | "CODIFIES" | "INTERPRETS"
// Matches the backend is_good_law() status values exactly (see
// backend/services/citation_graph.py): "good" | "bad" | "caution" | "unknown".
export type GoodLawStatus = "good" | "bad" | "caution" | "unknown"

// Honest per-node validity from /matters/{id}/graph:
//   "bad"        — has an incoming negative treatment edge (overruled etc.)
//   "good"       — authoritatively confirmed good law (CourtListener-verified)
//   "no_adverse" — a real authority with NO adverse treatment found in this graph
//   "unknown"    — provenance/document node or undetermined
export type NodeGoodLawStatus = "bad" | "good" | "no_adverse" | "unknown"

export interface GraphNode {
  id: string; citation: string; name?: string; type: "case" | "statute" | "regulation" | "document"
  year?: number; jurisdiction?: string; court?: string; authority_score?: number
  is_good_law?: boolean | null; is_verified?: boolean
  good_law_status?: NodeGoodLawStatus
}

export interface GraphEdge {
  source: string; target: string; treatment: TreatmentType; confidence?: number
}

export interface CitationNetwork {
  nodes: GraphNode[]; edges: GraphEdge[]
  stats?: { total_nodes: number; total_edges: number }
}

// Matches the backend is_good_law() return dict exactly.
export interface GoodLawNegativeTreatment {
  treatment: TreatmentType
  citing_citation: string
  citing_name: string | null
  context: string | null
}

export interface GoodLawResponse {
  status: GoodLawStatus
  is_good_law: boolean | null
  negative_treatments: GoodLawNegativeTreatment[]
  positive_count: number
  cautionary_count: number
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

// ============================================
// Conversation Types
// ============================================

export interface ConversationItem {
  id: string
  title: string | null
  created_at: string
  updated_at: string
  message_count: number
  last_message_preview: string | null
}

export interface ConversationQueryItem {
  id: string
  question: string
  answer: string
  citations: AskSourceItem[] | null
  citation_verification?: RawCitationVerification | null
  claim_verification?: RawClaimVerification | null
  conflict_analysis?: RawConflictAnalysis | null
  created_at: string
}

export interface AskSourceItem {
  chunk_id: string
  page_num: string
  section_name: string
  relevance_score: number
  content: string
  document_id: string
  document_name: string
  source_type?: "document" | "case_law"
  url?: string
  court_level?: string
  jurisdiction_code?: string
  authority_score?: number
  binding_authority?: boolean
  effective_date?: string | null
  document_status?: string
}

export interface ConversationDetail {
  id: string
  title: string | null
  created_at: string
  updated_at: string
  queries: ConversationQueryItem[]
}
