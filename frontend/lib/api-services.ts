import api from "./api"
import type {
  ChunkResponse,
  ContractReviewResult,
  DraftResponse,
  AuditLogEntry,
  PrecedentSearchResult,
  SavedPrecedent,
  ConversationItem,
  GraphNode,
  GraphEdge,
  CitationNetwork,
  TreatmentType,
  GoodLawResponse,
} from "./types"

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

// ============================================
// Raw (snake_case) sub-shapes returned by the backend /ask endpoint.
// Exported so types.ts can reuse them for ConversationQueryItem.
// ============================================

export interface RawVerifiedCitation {
  index: number
  raw_text: string
  case_name?: string
  court?: string
  date?: string
  jurisdiction?: string
  status: "verified" | "partial" | "unverified" | "not_found" | "unverifiable"
  case_status?: "good_law" | "caution" | "bad_law" | "unknown"
  confidence: number
  source_url?: string
  citation_count?: number
  quote_match?: number
  matched_source_idx?: number
  verification_method?: string
}

export interface RawCitationVerification {
  annotated_answer: string
  verified_citations: RawVerifiedCitation[]
  summary: {
    total: number
    verified: number
    partial: number
    unverified: number
    not_found: number
    unverifiable: number
  }
}

export interface RawVerifiedClaim {
  claim_text: string
  cited_sources: number[]
  status: "supported" | "partially_supported" | "unsupported" | "uncited"
  confidence: number
  verification_tier: number
  issues: string[]
  source_excerpt?: string
  requires_review?: boolean
}

export interface RawClaimVerification {
  verified_claims: RawVerifiedClaim[]
  summary: {
    total: number
    supported: number
    partially_supported: number
    unsupported: number
    uncited: number
  }
}

export interface RawCredibility {
  score: number
  authority_type: string
  authority_weight: number
  recency_weight: number
  specificity_weight: number
  citation_weight: number
}

export interface RawConflictChunk {
  document_id: string
  document_name: string
  page_num: string
  section_name?: string
  content_snippet: string
  credibility: RawCredibility
}

export interface RawConflict {
  id: string
  severity: "high" | "medium"
  contradiction_score: number
  chunk_a: RawConflictChunk
  chunk_b: RawConflictChunk
  recommended_source: "a" | "b"
  explanation: string
}

export interface RawConflictAnalysis {
  has_conflicts: boolean
  conflicts: RawConflict[]
  summary: {
    total_conflicts: number
    high_severity: number
    medium_severity: number
    recommended_action: string
  }
}

export interface RawIssueItem {
  domain: string
  issue: string
  legal_question: string
  confidence: number
  key_facts: string[]
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
    source_type?: "document" | "case_law"
    url?: string
    court_level?: string
    jurisdiction_code?: string
    authority_score?: number
    binding_authority?: boolean
    effective_date?: string | null
    document_status?: string
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
  citation_verification?: RawCitationVerification | null
  claim_verification?: RawClaimVerification | null
  conflict_analysis?: RawConflictAnalysis | null
  issue_analysis?: {
    issues: {
      domain: string
      issue: string
      legal_question: string
      confidence: number
      key_facts: string[]
    }[]
    missing_information: string[]
  } | null
  error: string | null
  query_id?: string
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

export async function askQuestion(
  matterId: string,
  question: string,
  includeLegalResearch: boolean = false,
  conversationId?: string,
): Promise<AskResponse> {
  const { data } = await api.post<AskResponse>(`/matters/${matterId}/ask`, {
    question,
    include_legal_research: includeLegalResearch,
    conversation_id: conversationId ?? null,
  })
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
// Query Delete API Functions
// ============================================

export async function deleteAllQueries(
  matterId: string
): Promise<{ matter_id: string; deleted_count: number }> {
  const { data } = await api.delete(`/matters/${matterId}/queries`)
  return data
}

export async function deleteQuery(
  matterId: string,
  queryId: string
): Promise<{ id: string; deleted: boolean }> {
  const { data } = await api.delete(`/matters/${matterId}/queries/${queryId}`)
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

// ============================================
// Contract Review API Functions
// ============================================

export async function getContractReview(
  matterId: string,
  documentId?: string
): Promise<ContractReviewResult> {
  const params = documentId ? { document_id: documentId } : {}
  const { data } = await api.get<ContractReviewResult>(
    `/matters/${matterId}/contract-review`,
    { params }
  )
  return data
}

export async function runContractReview(
  matterId: string,
  documentId?: string
): Promise<ContractReviewResult> {
  const { data } = await api.post<ContractReviewResult>(
    `/matters/${matterId}/contract-review`,
    { document_id: documentId || null }
  )
  return data
}

// ============================================
// Draft Assistant API Functions
// ============================================

export async function createDraft(
  matterId: string,
  documentType: string,
  instructions: string
): Promise<DraftResponse> {
  const { data } = await api.post<DraftResponse>(
    `/matters/${matterId}/drafts`,
    { document_type: documentType, instructions }
  )
  return data
}

export async function listDrafts(matterId: string): Promise<DraftResponse[]> {
  const { data } = await api.get<DraftResponse[]>(`/matters/${matterId}/drafts`)
  return data
}

// ============================================
// Audit Log API Functions
// ============================================

export async function getAuditLog(
  matterId: string,
  limit = 100
): Promise<AuditLogEntry[]> {
  const { data } = await api.get<AuditLogEntry[]>(
    `/matters/${matterId}/audit-log`,
    { params: { limit } }
  )
  return data
}

// ============================================
// Precedent API Functions
// ============================================

export async function searchPrecedents(
  query: string
): Promise<{ results: PrecedentSearchResult[]; total: number }> {
  const { data } = await api.post<{
    results: PrecedentSearchResult[]
    total: number
  }>("/precedents/search", { query })
  return data
}

export async function savePrecedent(
  precedent: Omit<SavedPrecedent, "id" | "created_at">
): Promise<{ id: string; title: string; created_at: string }> {
  const { data } = await api.post<{
    id: string
    title: string
    created_at: string
  }>("/precedents/save", precedent)
  return data
}

export async function listPrecedents(): Promise<SavedPrecedent[]> {
  const { data } = await api.get<SavedPrecedent[]>("/precedents")
  return data
}

export async function deletePrecedent(
  id: string
): Promise<{ id: string; deleted: boolean }> {
  const { data } = await api.delete(`/precedents/${id}`)
  return data
}

// ============================================
// Conversation API Functions
// ============================================

export async function listConversations(matterId: string): Promise<ConversationItem[]> {
  const { data } = await api.get<ConversationItem[]>(`/matters/${matterId}/conversations`)
  return data
}

export async function getConversation(
  matterId: string,
  conversationId: string
): Promise<import("./types").ConversationDetail> {
  const { data } = await api.get<import("./types").ConversationDetail>(
    `/matters/${matterId}/conversations/${conversationId}`
  )
  return data
}

export async function createConversation(
  matterId: string
): Promise<{ id: string; title: string | null; created_at: string }> {
  const { data } = await api.post<{ id: string; title: string | null; created_at: string }>(
    `/matters/${matterId}/conversations`
  )
  return data
}

export async function deleteConversation(
  matterId: string,
  conversationId: string
): Promise<{ id: string; deleted: boolean }> {
  const { data } = await api.delete(`/matters/${matterId}/conversations/${conversationId}`)
  return data
}

export async function updateConversationTitle(
  matterId: string,
  conversationId: string,
  title: string
): Promise<ConversationItem> {
  const { data } = await api.patch<ConversationItem>(
    `/matters/${matterId}/conversations/${conversationId}`,
    { title }
  )
  return data
}

// ============================================
// Citation Knowledge Graph API Functions
// ============================================

// Raw shapes returned by the backend citation_graph serializers
// (_node_to_dict / _edge_to_dict use snake_case DB field names).
interface RawGraphNode {
  id: string
  node_type: GraphNode["type"]
  citation_text: string
  name?: string | null
  court?: string | null
  year?: number | null
  jurisdiction?: string | null
  authority_score?: number | null
  is_verified?: boolean
  is_good_law?: boolean | null
  good_law_status?: "bad" | "good" | "no_adverse" | "unknown"
}

interface RawGraphEdge {
  source_id: string
  target_id: string
  treatment: TreatmentType
  confidence?: number | null
}

interface RawCitationNetwork {
  nodes?: RawGraphNode[]
  edges?: RawGraphEdge[]
  stats?: { total_nodes: number; total_edges: number }
}

function mapGraphNode(n: RawGraphNode): GraphNode {
  return {
    id: n.id,
    citation: n.citation_text,
    name: n.name ?? undefined,
    type: n.node_type,
    year: n.year ?? undefined,
    jurisdiction: n.jurisdiction ?? undefined,
    court: n.court ?? undefined,
    authority_score: n.authority_score ?? undefined,
    is_good_law: n.is_good_law ?? null,
    is_verified: n.is_verified,
    good_law_status: n.good_law_status,
  }
}

function mapGraphEdge(e: RawGraphEdge): GraphEdge {
  return {
    source: e.source_id,
    target: e.target_id,
    treatment: e.treatment,
    confidence: e.confidence ?? undefined,
  }
}

function normalizeCitationNetwork(raw: RawCitationNetwork | null | undefined): CitationNetwork {
  // Distinguish a genuinely empty graph (e.g. `{ nodes: [] }`) from a malformed
  // response. Only treat it as malformed when the payload is missing entirely or
  // carries neither a `nodes` nor an `edges` field.
  if (raw == null || (raw.nodes === undefined && raw.edges === undefined)) {
    throw new Error("Malformed citation graph response")
  }
  return {
    nodes: (raw.nodes ?? []).map(mapGraphNode),
    edges: (raw.edges ?? []).map(mapGraphEdge),
    stats: raw.stats,
  }
}

async function parseGraphJson(res: Response): Promise<RawCitationNetwork> {
  try {
    return (await res.json()) as RawCitationNetwork
  } catch {
    throw new Error("Citation graph response was not valid JSON")
  }
}

export async function getGoodLawStatus(citation: string): Promise<GoodLawResponse> {
  const res = await fetch(
    `${api.defaults.baseURL}/graph/case/${encodeURIComponent(citation)}/good-law`
  )
  if (!res.ok) throw new Error("Failed to check good law status")
  return res.json() as Promise<GoodLawResponse>
}

export async function getCitationNetwork(
  citation: string,
  depth = 2,
  maxNodes = 100
): Promise<CitationNetwork> {
  const res = await fetch(
    `${api.defaults.baseURL}/graph/case/${encodeURIComponent(citation)}/network?depth=${depth}&max_nodes=${maxNodes}`
  )
  if (!res.ok) throw new Error("Failed to get citation network")
  return normalizeCitationNetwork(await parseGraphJson(res))
}

export async function getMatterGraph(matterId: string): Promise<CitationNetwork> {
  const res = await fetch(`${api.defaults.baseURL}/matters/${matterId}/graph`)
  if (!res.ok) throw new Error("Failed to get matter graph")
  return normalizeCitationNetwork(await parseGraphJson(res))
}
