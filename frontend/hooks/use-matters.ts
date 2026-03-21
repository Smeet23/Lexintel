"use client"

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import {
  listMatters,
  getMatter,
  createMatter,
  deleteMatter,
  cancelMatterProcessing,
  askQuestion,
  listMatterDocuments,
  uploadMatterDocument,
  fetchMatterChunks,
  getQueryHistory,
  deleteDocument,
  deleteAllQueries,
  deleteQuery,
  getContractReview,
  runContractReview,
  createDraft,
  listDrafts,
  getAuditLog,
  searchPrecedents,
  savePrecedent,
  listPrecedents,
  deletePrecedent,
  listConversations,
  createConversation,
  deleteConversation,
  type MatterResponse,
  type MatterDetailResponse,
  type AskResponse,
  type DocumentResponse,
  type QueryHistoryItem,
} from "@/lib/api-services"
import type {
  ChunkResponse,
  ContractReviewResult,
  DraftResponse,
  AuditLogEntry,
  PrecedentSearchResult,
  SavedPrecedent,
  ConversationItem,
} from "@/lib/types"

export function useMatters() {
  return useQuery<MatterResponse[]>({
    queryKey: ["matters"],
    queryFn: listMatters,
    refetchInterval: 10_000, // poll every 10s for status updates
  })
}

export function useMatter(id: string) {
  return useQuery<MatterDetailResponse>({
    queryKey: ["matters", id],
    queryFn: () => getMatter(id),
    enabled: !!id,
  })
}

export function useCreateMatter() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({
      name,
      files,
      signal,
    }: {
      name: string
      files: File[]
      signal?: AbortSignal
    }) => createMatter(name, files, signal),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["matters"] })
    },
  })
}

export function useDeleteMatter() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (id: string) => deleteMatter(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["matters"] })
    },
  })
}

export function useCancelMatterProcessing(matterId: string) {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: () => cancelMatterProcessing(matterId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["matters"] })
      queryClient.invalidateQueries({ queryKey: ["matters", matterId] })
    },
  })
}

export function useAskQuestion(matterId: string) {
  const queryClient = useQueryClient()

  return useMutation<AskResponse, Error, { question: string; includeLegalResearch?: boolean; conversationId?: string }>({
    mutationFn: ({ question, includeLegalResearch, conversationId }) =>
      askQuestion(matterId, question, includeLegalResearch ?? false, conversationId),
    onSuccess: (_data, variables) => {
      queryClient.invalidateQueries({ queryKey: ["matters", matterId] })
      queryClient.invalidateQueries({ queryKey: ["matters", matterId, "queries"] })
      if (variables.conversationId) {
        queryClient.invalidateQueries({ queryKey: ["matters", matterId, "conversations"] })
      }
    },
  })
}

export function useMatterDocuments(matterId: string) {
  return useQuery<DocumentResponse[]>({
    queryKey: ["matters", matterId, "documents"],
    queryFn: () => listMatterDocuments(matterId),
    enabled: !!matterId,
    refetchInterval: 10_000, // poll for processing status updates
  })
}

export function useMatterChunks(matterId: string, documentId?: string) {
  return useQuery<ChunkResponse[]>({
    queryKey: ["matters", matterId, "chunks", documentId ?? "all"],
    queryFn: () => fetchMatterChunks(matterId, documentId),
    enabled: !!matterId,
    staleTime: 5 * 60 * 1000, // chunks don't change once processed
  })
}

export function useQueryHistory(matterId: string) {
  return useQuery<QueryHistoryItem[]>({
    queryKey: ["matters", matterId, "queries"],
    queryFn: () => getQueryHistory(matterId),
    enabled: !!matterId,
    staleTime: 30_000,
  })
}

export function useDeleteAllQueries(matterId: string) {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: () => deleteAllQueries(matterId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["matters", matterId, "queries"] })
      queryClient.invalidateQueries({ queryKey: ["matters", matterId] })
      queryClient.invalidateQueries({ queryKey: ["matters", matterId, "audit-log"] })
      queryClient.invalidateQueries({ queryKey: ["matters", matterId, "conversations"] })
    },
  })
}

export function useDeleteQuery(matterId: string) {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (queryId: string) => deleteQuery(matterId, queryId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["matters", matterId, "queries"] })
      queryClient.invalidateQueries({ queryKey: ["matters", matterId] })
      queryClient.invalidateQueries({ queryKey: ["matters", matterId, "audit-log"] })
    },
  })
}

export function useDeleteDocument(matterId: string) {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (documentId: string) => deleteDocument(matterId, documentId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["matters", matterId, "documents"] })
      queryClient.invalidateQueries({ queryKey: ["matters", matterId] })
      queryClient.invalidateQueries({ queryKey: ["matters"] })
    },
  })
}

export function useUploadMatterDocument(matterId: string) {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ file, signal }: { file: File; signal?: AbortSignal }) =>
      uploadMatterDocument(matterId, file, signal),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["matters", matterId, "documents"] })
      queryClient.invalidateQueries({ queryKey: ["matters", matterId] })
      queryClient.invalidateQueries({ queryKey: ["matters"] })
    },
  })
}

// ============================================
// Contract Review Hooks
// ============================================

export function useContractReview(matterId: string, documentId?: string) {
  return useQuery<ContractReviewResult>({
    queryKey: ["matters", matterId, "contract-review", documentId ?? "latest"],
    queryFn: () => getContractReview(matterId, documentId),
    enabled: !!matterId,
    staleTime: 5 * 60 * 1000,
  })
}

export function useRunContractReview(matterId: string) {
  const queryClient = useQueryClient()

  return useMutation<ContractReviewResult, Error, string | undefined>({
    mutationFn: (documentId?: string) => runContractReview(matterId, documentId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["matters", matterId, "contract-review"] })
      queryClient.invalidateQueries({ queryKey: ["matters", matterId, "audit-log"] })
    },
  })
}

// ============================================
// Draft Hooks
// ============================================

export function useDrafts(matterId: string) {
  return useQuery<DraftResponse[]>({
    queryKey: ["matters", matterId, "drafts"],
    queryFn: () => listDrafts(matterId),
    enabled: !!matterId,
  })
}

export function useCreateDraft(matterId: string) {
  const queryClient = useQueryClient()

  return useMutation<
    DraftResponse,
    Error,
    { documentType: string; instructions: string }
  >({
    mutationFn: ({ documentType, instructions }) =>
      createDraft(matterId, documentType, instructions),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["matters", matterId, "drafts"] })
      queryClient.invalidateQueries({ queryKey: ["matters", matterId, "audit-log"] })
    },
  })
}

// ============================================
// Audit Log Hooks
// ============================================

export function useAuditLog(matterId: string) {
  return useQuery<AuditLogEntry[]>({
    queryKey: ["matters", matterId, "audit-log"],
    queryFn: () => getAuditLog(matterId),
    enabled: !!matterId,
    refetchInterval: 30_000,
  })
}

// ============================================
// Precedent Hooks
// ============================================

export function usePrecedentSearch(query: string | null) {
  return useQuery<{ results: PrecedentSearchResult[]; total: number }>({
    queryKey: ["precedents", "search", query],
    queryFn: () => searchPrecedents(query!),
    enabled: query !== null && query.length >= 3,
    placeholderData: (prev) => prev,
    staleTime: 2 * 60 * 1000,
  })
}

export function useSavedPrecedents() {
  return useQuery<SavedPrecedent[]>({
    queryKey: ["precedents", "saved"],
    queryFn: listPrecedents,
  })
}

export function useSavePrecedent() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (precedent: Omit<SavedPrecedent, "id" | "created_at">) =>
      savePrecedent(precedent),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["precedents", "saved"] })
    },
  })
}

export function useDeletePrecedent() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (id: string) => deletePrecedent(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["precedents", "saved"] })
    },
  })
}

// ============================================
// Conversation Hooks
// ============================================

export function useConversations(matterId: string) {
  return useQuery<ConversationItem[]>({
    queryKey: ["matters", matterId, "conversations"],
    queryFn: () => listConversations(matterId),
    enabled: !!matterId,
    staleTime: 10_000,
  })
}

export function useCreateConversation(matterId: string) {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: () => createConversation(matterId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["matters", matterId, "conversations"] })
    },
  })
}

export function useDeleteConversation(matterId: string) {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (conversationId: string) => deleteConversation(matterId, conversationId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["matters", matterId, "conversations"] })
    },
  })
}

