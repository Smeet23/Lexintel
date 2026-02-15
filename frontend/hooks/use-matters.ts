"use client"

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import {
  listMatters,
  getMatter,
  createMatter,
  deleteMatter,
  askQuestion,
  type MatterResponse,
  type MatterDetailResponse,
  type AskResponse,
} from "@/lib/api-services"

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
    mutationFn: ({ name, file }: { name: string; file: File }) =>
      createMatter(name, file),
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

export function useAskQuestion(matterId: string) {
  const queryClient = useQueryClient()

  return useMutation<AskResponse, Error, string>({
    mutationFn: (question: string) => askQuestion(matterId, question),
    onSuccess: () => {
      // Refresh matter data to update query count
      queryClient.invalidateQueries({ queryKey: ["matters", matterId] })
    },
  })
}
