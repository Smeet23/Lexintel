"use client"

import React, { useState, useCallback, useEffect, useRef } from "react"
import { useParams, useRouter } from "next/navigation"
import {
  ArrowLeft,
  MessageSquare,
  Shield,
  PenLine,
  ClipboardList,
  FileText,
  Loader2,
  AlertTriangle,
  Download,
  Square,
} from "lucide-react"
import { motion } from "framer-motion"
import AppLayout from "@/layouts/AppLayout"
import ChatPanel from "@/components/ChatPanel"
import CitationPanel from "@/components/CitationPanel"
import DocumentTab from "@/components/DocumentTab"
import DataTable from "@/components/DataTable"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Progress } from "@/components/ui/progress"
import { cn, formatRelativeTime } from "@/lib/utils"
import { useMatter, useAskQuestion, useCancelMatterProcessing, useQueryHistory } from "@/hooks/use-matters"
import { useMatterProgress } from "@/hooks/use-progress"
import type { QueryMessage, Citation, AuditEntry } from "@/lib/types"

// Mock audit log (no backend endpoint for this yet)
const mockAuditLog: (AuditEntry & Record<string, unknown>)[] = [
  { id: "a1", action: "Query", user: "John Smith", details: "Summarize key risks in this contract", sources: ["MSA.pdf - Page 12", "Amendment 3 - Page 4"], timestamp: new Date(Date.now() - 3600000).toISOString() },
  { id: "a2", action: "Upload", user: "John Smith", details: "Uploaded document", timestamp: new Date(Date.now() - 7200000).toISOString() },
]

// Contract review mock (no backend endpoint for this yet)
const contractRisks = [
  { clause: "Indemnification (Section 8.2)", risk: "high" as const, summary: "Unlimited indemnification exposure for IP infringement claims. No cap or basket provisions." },
  { clause: "Termination for Convenience (Section 12.1)", risk: "medium" as const, summary: "30-day notice period may be insufficient. No wind-down provisions for ongoing work." },
  { clause: "Limitation of Liability (Section 9.1)", risk: "low" as const, summary: "Standard mutual cap at 12 months' fees. Carve-outs are appropriate." },
]

export default function MatterWorkspacePage() {
  const params = useParams()
  const router = useRouter()
  const matterId = params.id as string

  const [messages, setMessages] = useState<QueryMessage[]>([])
  const [selectedCitations, setSelectedCitations] = useState<Citation[]>([])
  const [chunkPreview, setChunkPreview] = useState<Citation | null>(null)
  const [draftType, setDraftType] = useState("")

  const { data: matter, isLoading, error } = useMatter(matterId)
  const { data: queryHistory } = useQueryHistory(matterId)
  const askQuestion = useAskQuestion(matterId)
  const cancelProcessing = useCancelMatterProcessing(matterId)
  const { progress } = useMatterProgress(matterId, matter?.status === "processing")
  const historyRestoredRef = useRef(false)

  // Load conversation history on mount (runs exactly once when both data are ready)
  useEffect(() => {
    if (historyRestoredRef.current || !queryHistory?.length || !matter) return
    historyRestoredRef.current = true

    const restored: QueryMessage[] = []
    for (const item of queryHistory) {
      restored.push({
        id: `hist-q-${item.id}`,
        role: "user",
        content: item.question,
        timestamp: item.created_at,
      })

      const citations: Citation[] = (item.citations || []).map((s) => ({
        documentName: (s as any).document_name || matter.name,
        pageNumber: parseInt(s.page_num) || 0,
        section: (s as any).section_name || "",
        excerpt: s.content?.slice(0, 200) || "",
        relevanceScore: s.relevance_score || 0,
        content: s.content ?? undefined,
      }))

      restored.push({
        id: `hist-a-${item.id}`,
        role: "assistant",
        content: item.answer,
        citations: citations.length > 0 ? citations : undefined,
        timestamp: item.created_at,
      })
    }
    setMessages(restored)
  }, [queryHistory, matter])

  const handleSendMessage = useCallback((content: string) => {
    const userMsg: QueryMessage = {
      id: `msg-${Date.now()}`,
      role: "user",
      content,
      timestamp: new Date().toISOString(),
    }
    setMessages((prev) => [...prev, userMsg])

    askQuestion.mutate(content, {
      onSuccess: (result) => {
        if (result.answer) {
          // Map backend sources to Citation format
          const citations: Citation[] = (result.sources || []).map((s) => ({
            documentName: s.document_name || matter?.name || "Document",
            pageNumber: parseInt(s.page_num) || 0,
            section: s.section_name || "",
            excerpt: s.content?.slice(0, 200) || "",
            relevanceScore: s.relevance_score || 0,
            content: s.content ?? undefined,
          }))

          const confidenceScore = typeof result.confidence === "object"
            ? Math.round((result.confidence.score || 0) * 100)
            : 0

          const aiMsg: QueryMessage = {
            id: `msg-${Date.now() + 1}`,
            role: "assistant",
            content: result.answer,
            citations,
            confidenceScore,
            timestamp: new Date().toISOString(),
          }
          setMessages((prev) => [...prev, aiMsg])
          setSelectedCitations(citations)
        } else {
          const errorMsg: QueryMessage = {
            id: `msg-${Date.now() + 1}`,
            role: "assistant",
            content: result.error || "Sorry, I couldn't generate an answer. Please try rephrasing your question.",
            timestamp: new Date().toISOString(),
          }
          setMessages((prev) => [...prev, errorMsg])
        }
      },
      onError: () => {
        const errorMsg: QueryMessage = {
          id: `msg-${Date.now() + 1}`,
          role: "assistant",
          content: "An error occurred while processing your question. Please try again.",
          timestamp: new Date().toISOString(),
        }
        setMessages((prev) => [...prev, errorMsg])
      },
    })
  }, [askQuestion, matter])

  const auditColumns = [
    {
      key: "action",
      header: "Action",
      render: (item: typeof mockAuditLog[0]) => (
        <Badge variant={item.action === "Query" ? "default" : item.action === "Upload" ? "active" : "review"}>
          {item.action as string}
        </Badge>
      ),
    },
    {
      key: "user",
      header: "User",
      render: (item: typeof mockAuditLog[0]) => <span className="text-sm font-medium">{item.user}</span>,
    },
    {
      key: "details",
      header: "Details",
      render: (item: typeof mockAuditLog[0]) => <span className="text-sm text-muted">{item.details}</span>,
    },
    {
      key: "sources",
      header: "Sources",
      render: (item: typeof mockAuditLog[0]) => (
        <span className="text-xs text-muted">
          {(item.sources as string[] | undefined)?.join(", ") || "-"}
        </span>
      ),
    },
    {
      key: "timestamp",
      header: "Time",
      render: (item: typeof mockAuditLog[0]) => <span className="text-xs text-muted">{formatRelativeTime(item.timestamp)}</span>,
    },
  ]

  if (isLoading) {
    return (
      <AppLayout title="Loading...">
        <div className="flex items-center justify-center py-24">
          <Loader2 className="h-6 w-6 animate-spin text-muted" />
          <span className="ml-2 text-muted">Loading matter...</span>
        </div>
      </AppLayout>
    )
  }

  if (error || !matter) {
    return (
      <AppLayout title="Error">
        <div className="py-24 text-center">
          <p className="text-muted">Matter not found or backend unavailable.</p>
          <Button variant="outline" className="mt-4" onClick={() => router.push("/matters")}>
            <ArrowLeft className="h-4 w-4" /> Back to Matters
          </Button>
        </div>
      </AppLayout>
    )
  }

  const statusBadgeVariant = matter.status === "ready" ? "active" : matter.status === "processing" ? "review" : "error"
  const statusLabel = matter.status === "ready" ? "Ready" : matter.status === "processing" ? "Processing" : matter.status === "cancelled" ? "Cancelled" : "Error"

  return (
    <AppLayout title={matter.name}>
      {/* Breadcrumb + Header */}
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
        className="mb-8"
      >
        <button
          onClick={() => router.push("/matters")}
          className="flex items-center gap-1 text-sm text-muted hover:text-foreground transition-colors mb-4 cursor-pointer"
        >
          <ArrowLeft className="h-4 w-4" /> Back to Matters
        </button>

        <div className="flex items-start justify-between">
          <div>
            <div className="flex items-center gap-3">
              <h2 className="text-2xl font-display font-semibold text-foreground">{matter.name}</h2>
              <Badge variant={statusBadgeVariant}>{statusLabel}</Badge>
            </div>
            <p className="text-sm text-muted mt-1.5">
              {matter.file_type.toUpperCase()} &middot; {matter.documents_count} chunks &middot; {matter.queries_count} queries &middot; Created {formatRelativeTime(matter.created_at)}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm">
              <Download className="h-4 w-4" />
              Export Bundle
            </Button>
          </div>
        </div>
      </motion.div>

      {/* Processing indicator with real-time progress */}
      {matter.status === "processing" && (
        <div className="mb-6 rounded-xl border border-amber-200/60 bg-amber-50/60 p-4 space-y-3">
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-3">
              <Loader2 className="h-5 w-5 animate-spin text-amber-600 shrink-0" />
              <div>
                <p className="text-sm font-medium text-amber-800">
                  {progress?.message || "Document is being processed"}
                </p>
                <p className="text-xs text-amber-600 mt-0.5">
                  {progress?.detail
                    ? `${progress.detail} — Step ${progress.step} of ${progress.total_steps}`
                    : "Connecting to processing pipeline..."}
                </p>
              </div>
            </div>
            <Button
              variant="outline"
              size="sm"
              className="shrink-0 border-destructive/50 text-destructive hover:bg-destructive/10"
              onClick={() => cancelProcessing.mutate()}
              disabled={cancelProcessing.isPending}
            >
              {cancelProcessing.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <>
                  <Square className="h-4 w-4 mr-1.5" />
                  Cancel
                </>
              )}
            </Button>
          </div>

          {/* Progress bar */}
          {progress && (
            <div className="space-y-1.5">
              <div className="flex items-center justify-between text-xs text-amber-700">
                <span className="capitalize">{progress.stage}</span>
                <span>{Math.round(((progress.step - 1) / progress.total_steps) * 100 + (progress.progress / progress.total_steps))}%</span>
              </div>
              <div className="h-2 w-full rounded-full bg-amber-200/60 overflow-hidden">
                <div
                  className="h-full rounded-full bg-amber-500 transition-all duration-500 ease-out"
                  style={{
                    width: `${Math.round(((progress.step - 1) / progress.total_steps) * 100 + (progress.progress / progress.total_steps))}%`,
                  }}
                />
              </div>
            </div>
          )}
        </div>
      )}

      {/* Tabs */}
      <Tabs defaultValue="ask-ai" className="space-y-6">
        <TabsList>
          <TabsTrigger value="ask-ai">
            <MessageSquare className="h-4 w-4 mr-2" />
            Ask AI
          </TabsTrigger>
          <TabsTrigger value="documents">
            <FileText className="h-4 w-4 mr-2" />
            Documents
          </TabsTrigger>
          <TabsTrigger value="contract-review">
            <Shield className="h-4 w-4 mr-2" />
            Contract Review
          </TabsTrigger>
          <TabsTrigger value="draft-assistant">
            <PenLine className="h-4 w-4 mr-2" />
            Draft Assistant
          </TabsTrigger>
          <TabsTrigger value="audit-log">
            <ClipboardList className="h-4 w-4 mr-2" />
            Audit Log
          </TabsTrigger>
        </TabsList>

        {/* Ask AI Tab */}
        <TabsContent value="ask-ai">
          <div className="flex gap-6 h-[calc(100vh-280px)]">
            <div className="flex-1 bg-white rounded-xl border border-border shadow-elevated overflow-hidden">
              <ChatPanel
                messages={messages}
                onSend={handleSendMessage}
                isLoading={askQuestion.isPending}
                onSelectCitation={setSelectedCitations}
                onCitationClick={setChunkPreview}
              />
            </div>
            <div className="w-80 bg-white rounded-xl border border-border shadow-elevated p-5 overflow-y-auto">
              <CitationPanel
                citations={selectedCitations}
                onCitationClick={setChunkPreview}
              />
            </div>
          </div>

          {/* Chunk preview dialog (clickable sources) */}
          <Dialog open={!!chunkPreview} onOpenChange={(open) => !open && setChunkPreview(null)}>
            <DialogContent className="max-w-2xl max-h-[85vh] flex flex-col">
              <DialogHeader>
                <DialogTitle className="text-base">
                  {chunkPreview?.documentName}
                  {chunkPreview != null && (
                    <span className="text-muted-foreground font-normal ml-2">
                      Page {chunkPreview.pageNumber}
                      {chunkPreview.section && ` · ${chunkPreview.section}`}
                      <span className="ml-2 text-[11px] font-mono">
                        {Math.round((chunkPreview.relevanceScore ?? 0) * 100)}% relevance
                      </span>
                    </span>
                  )}
                </DialogTitle>
              </DialogHeader>
              <div className="overflow-y-auto flex-1 pr-2 text-[13px] leading-relaxed text-foreground whitespace-pre-wrap border border-border rounded-lg p-4 bg-surface/50">
                {chunkPreview?.content || chunkPreview?.excerpt || "No content."}
              </div>
            </DialogContent>
          </Dialog>
        </TabsContent>

        {/* Documents Tab */}
        <TabsContent value="documents">
          <DocumentTab matterId={matterId} />
        </TabsContent>

        {/* Contract Review Tab */}
        <TabsContent value="contract-review">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 space-y-4">
              <div className="bg-white rounded-xl border border-border shadow-elevated p-6">
                <h3 className="text-lg font-display font-semibold text-foreground mb-4">Risk Analysis</h3>
                <div className="space-y-3">
                  {contractRisks.map((risk, idx) => (
                    <div
                      key={idx}
                      className={cn(
                        "rounded-xl border p-4 transition-colors hover:shadow-sm cursor-pointer",
                        risk.risk === "high" ? "border-red-200 bg-red-50/60" :
                        risk.risk === "medium" ? "border-amber-200 bg-amber-50/60" :
                        "border-emerald-200 bg-emerald-50/60"
                      )}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex items-start gap-3">
                          <AlertTriangle className={cn(
                            "h-5 w-5 mt-0.5 shrink-0",
                            risk.risk === "high" ? "text-red-600" :
                            risk.risk === "medium" ? "text-amber-600" :
                            "text-emerald-600"
                          )} />
                          <div>
                            <p className="font-medium text-foreground text-sm">{risk.clause}</p>
                            <p className="text-sm text-muted mt-1">{risk.summary}</p>
                          </div>
                        </div>
                        <Badge variant={risk.risk === "high" ? "error" : risk.risk === "medium" ? "review" : "active"}>
                          {risk.risk.charAt(0).toUpperCase() + risk.risk.slice(1)} Risk
                        </Badge>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="space-y-4">
              <div className="bg-white rounded-xl border border-border shadow-elevated p-6">
                <h4 className="font-display font-semibold text-foreground mb-4">Summary</h4>
                <div className="space-y-3">
                  <div className="flex justify-between text-sm">
                    <span className="text-muted">Total Clauses Analyzed</span>
                    <span className="font-medium">24</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-muted">High Risk</span>
                    <span className="font-medium text-red-700">1</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-muted">Medium Risk</span>
                    <span className="font-medium text-amber-700">1</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-muted">Low Risk</span>
                    <span className="font-medium text-emerald-700">1</span>
                  </div>
                  <div className="border-t border-border pt-3">
                    <div className="flex justify-between text-sm mb-2">
                      <span className="text-muted">Overall Score</span>
                      <span className="font-bold text-amber-700">62/100</span>
                    </div>
                    <Progress value={62} />
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-xl border border-border shadow-elevated p-6">
                <h4 className="font-display font-semibold text-foreground mb-3">Missing Clauses</h4>
                <ul className="space-y-2 text-sm">
                  {["Force Majeure", "Non-Compete", "Audit Rights"].map((clause) => (
                    <li key={clause} className="flex items-center gap-2 text-muted">
                      <div className="h-1.5 w-1.5 rounded-full bg-foreground" />
                      {clause}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </TabsContent>

        {/* Draft Assistant Tab */}
        <TabsContent value="draft-assistant">
          <div className="max-w-3xl">
            <div className="bg-white rounded-xl border border-border shadow-elevated p-6">
              <h3 className="text-lg font-display font-semibold text-foreground mb-2">Draft Assistant</h3>
              <p className="text-sm text-muted mb-6">
                Generate legal documents with inline source references from your matter documents.
              </p>

              <div className="space-y-5">
                <div>
                  <label className="block text-sm font-medium text-foreground mb-2">Document Type</label>
                  <Select value={draftType} onValueChange={setDraftType}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select document type" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="brief">Legal Brief</SelectItem>
                      <SelectItem value="memo">Legal Memorandum</SelectItem>
                      <SelectItem value="motion">Motion</SelectItem>
                      <SelectItem value="response">Response Letter</SelectItem>
                      <SelectItem value="summary">Executive Summary</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-foreground mb-2">Instructions</label>
                  <textarea
                    placeholder="Describe what you need drafted. Be specific about the audience, key points to cover, and any constraints..."
                    className="flex min-h-[120px] w-full rounded-lg border border-input bg-white px-3 py-2 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-foreground/10 focus:border-foreground/20 resize-none"
                  />
                </div>

                <Button disabled={!draftType}>
                  <PenLine className="h-4 w-4" />
                  Generate Draft
                </Button>
              </div>
            </div>
          </div>
        </TabsContent>

        {/* Audit Log Tab */}
        <TabsContent value="audit-log">
          <div className="bg-white rounded-xl border border-border shadow-elevated">
            <div className="p-4 border-b border-border flex items-center justify-between">
              <h3 className="font-display font-semibold text-foreground">Activity Log</h3>
              <Button variant="outline" size="sm">
                <Download className="h-4 w-4" />
                Export Log
              </Button>
            </div>
            <DataTable columns={auditColumns} data={mockAuditLog} />
          </div>
        </TabsContent>
      </Tabs>
    </AppLayout>
  )
}
