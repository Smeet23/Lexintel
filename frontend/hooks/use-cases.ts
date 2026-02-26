import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { fetchCases, createCase, getCaseStatus, askQuestion, fetchCaseChunks } from "@/lib/api"

export function useCases() {
  return useQuery({
    queryKey: ["cases"],
    queryFn: fetchCases,
    refetchInterval: 10_000,
  })
}

export function useCreateCase() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ name, file }: { name: string; file: File }) =>
      createCase(name, file),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["cases"] })
    },
  })
}

export function useCaseStatus(caseId: string | undefined) {
  return useQuery({
    queryKey: ["case-status", caseId],
    queryFn: () => getCaseStatus(caseId!),
    enabled: !!caseId,
    refetchInterval: (query) => {
      const status = query.state.data?.status
      return status === "processing" ? 3_000 : false
    },
  })
}

export function useAskQuestion(caseId: string) {
  return useMutation({
    mutationFn: (question: string) => askQuestion(caseId, question),
  })
}

export function useCaseChunks(caseId: string) {
  return useQuery({
    queryKey: ["case-chunks", caseId],
    queryFn: () => fetchCaseChunks(caseId),
    enabled: !!caseId,
    staleTime: 5 * 60 * 1000, // 5 minutes — chunks don't change once processed
  })
}
