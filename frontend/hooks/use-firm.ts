"use client"

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import {
  getFirmTheme,
  updateFirmTheme,
  resetFirmTheme,
  getFirmMembers,
} from "@/lib/api-services"
import type { FirmThemeResponse, ThemeConfig, FirmMember } from "@/lib/types"

export function useFirmTheme(slug: string) {
  return useQuery<FirmThemeResponse>({
    queryKey: ["firm-theme", slug],
    queryFn: () => getFirmTheme(slug),
    staleTime: 5 * 60 * 1000,
    enabled: !!slug,
  })
}

export function useUpdateFirmTheme(slug: string) {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (theme: Partial<ThemeConfig>) => updateFirmTheme(slug, theme),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["firm-theme", slug] })
    },
  })
}

export function useResetFirmTheme(slug: string) {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: () => resetFirmTheme(slug),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["firm-theme", slug] })
    },
  })
}

export function useFirmMembers(slug: string) {
  return useQuery<FirmMember[]>({
    queryKey: ["firm-members", slug],
    queryFn: () => getFirmMembers(slug),
    enabled: !!slug,
  })
}
