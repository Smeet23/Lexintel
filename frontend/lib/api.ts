import axios from "axios"
import type { CaseResponse, CreateCaseResponse, CaseStatusResponse, RAGResponse, ChunkResponse } from "./types"

const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000",
  headers: {
    "Content-Type": "application/json",
  },
})

api.interceptors.request.use((config) => {
  if (typeof window !== "undefined") {
    const token = localStorage.getItem("veritas_token")
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
  }
  return config
})

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401 && typeof window !== "undefined") {
      localStorage.removeItem("veritas_token")
      window.location.href = "/login"
    }
    return Promise.reject(error)
  }
)

export default api

// ============================================
// API Service Functions
// ============================================

export async function fetchCases(): Promise<CaseResponse[]> {
  const { data } = await api.get<CaseResponse[]>("/cases")
  return data
}

export async function createCase(name: string, file: File): Promise<CreateCaseResponse> {
  const formData = new FormData()
  formData.append("name", name)
  formData.append("file", file)

  const { data } = await api.post<CreateCaseResponse>("/cases", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  })
  return data
}

export async function getCaseStatus(caseId: string): Promise<CaseStatusResponse> {
  const { data } = await api.get<CaseStatusResponse>(`/cases/${caseId}/status`)
  return data
}

export async function askQuestion(caseId: string, question: string): Promise<RAGResponse> {
  const { data } = await api.post<RAGResponse>(`/cases/${caseId}/ask`, { question })
  return data
}

export function subscribeToCaseProgress(caseId: string): EventSource {
  const baseURL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
  return new EventSource(`${baseURL}/cases/${caseId}/progress`)
}

export function getDocumentUrl(caseId: string): string {
  const baseURL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
  return `${baseURL}/cases/${caseId}/document`
}

export async function fetchCaseChunks(caseId: string): Promise<ChunkResponse[]> {
  const { data } = await api.get<ChunkResponse[]>(`/cases/${caseId}/chunks`)
  return data
}
