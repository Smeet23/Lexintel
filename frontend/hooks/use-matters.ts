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
  type MatterResponse,
  type MatterDetailResponse,
  type AskResponse,
  type DocumentResponse,
  type QueryHistoryItem,
} from "@/lib/api-services"
import type { ChunkResponse } from "@/lib/types"

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

  return useMutation<AskResponse, Error, string>({
    mutationFn: (question: string) => askQuestion(matterId, question),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["matters", matterId] })
      queryClient.invalidateQueries({ queryKey: ["matters", matterId, "queries"] })
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
