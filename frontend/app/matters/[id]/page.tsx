"use client"

import React, { useState, useCallback, useEffect, useRef } from "react"
import { useParams, useRouter } from "next/navigation"
import ReactMarkdown from "react-markdown"
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
import { useMatter, useAskQuestion, useCancelMatterProcessing, useQueryHistory, useContractReview, useRunContractReview, useDrafts, useCreateDraft, useAuditLog } from "@/hooks/use-matters"
import { useMatterProgress } from "@/hooks/use-progress"
import type { QueryMessage, Citation, AuditLogEntry } from "@/lib/types"

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
  const contractReview = useContractReview(matterId)
  const runContractReview = useRunContractReview(matterId)
  const drafts = useDrafts(matterId)
  const createDraft = useCreateDraft(matterId)
  const auditLog = useAuditLog(matterId)
  const [draftInstructions, setDraftInstructions] = useState("")
  const [selectedDraft, setSelectedDraft] = useState<import("@/lib/types").DraftResponse | null>(null)

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
        sourceType: ((s as any).source_type as "document" | "case_law") || "document",
        url: (s as any).url || undefined,
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

  const handleSendMessage = useCallback((content: string, includeLegalResearch: boolean) => {
    const userMsg: QueryMessage = {
      id: `msg-${Date.now()}`,
      role: "user",
      content,
      timestamp: new Date().toISOString(),
    }
    setMessages((prev) => [...prev, userMsg])

    askQuestion.mutate({ question: content, includeLegalResearch }, {
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
            sourceType: (s.source_type as "document" | "case_law") || "document",
            url: s.url || undefined,
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

        <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4">
          <div className="min-w-0">
            <div className="flex items-center gap-3 flex-wrap">
              <h2 className="text-xl sm:text-2xl font-display font-semibold text-foreground">{matter.name}</h2>
              <Badge variant={statusBadgeVariant}>{statusLabel}</Badge>
            </div>
            <p className="text-xs sm:text-sm text-muted mt-1.5">
              {matter.file_type.toUpperCase()} &middot; {matter.documents_count} chunks &middot; {matter.queries_count} queries &middot; Created {formatRelativeTime(matter.created_at)}
            </p>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            <Button variant="outline" size="sm">
              <Download className="h-4 w-4" />
              <span className="hidden sm:inline">Export Bundle</span>
              <span className="sm:hidden">Export</span>
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
        <div className="overflow-x-auto -mx-4 px-4 sm:mx-0 sm:px-0">
          <TabsList className="w-max sm:w-auto">
            <TabsTrigger value="ask-ai">
              <MessageSquare className="h-4 w-4 mr-1.5" />
              <span className="hidden sm:inline">Ask AI</span>
              <span className="sm:hidden">AI</span>
            </TabsTrigger>
            <TabsTrigger value="documents">
              <FileText className="h-4 w-4 mr-1.5" />
              <span className="hidden sm:inline">Documents</span>
              <span className="sm:hidden">Docs</span>
            </TabsTrigger>
            <TabsTrigger value="contract-review">
              <Shield className="h-4 w-4 mr-1.5" />
              <span className="hidden sm:inline">Contract Review</span>
              <span className="sm:hidden">Review</span>
            </TabsTrigger>
            <TabsTrigger value="draft-assistant">
              <PenLine className="h-4 w-4 mr-1.5" />
              <span className="hidden sm:inline">Draft Assistant</span>
              <span className="sm:hidden">Draft</span>
            </TabsTrigger>
            <TabsTrigger value="audit-log">
              <ClipboardList className="h-4 w-4 mr-1.5" />
              <span className="hidden sm:inline">Audit Log</span>
              <span className="sm:hidden">Audit</span>
            </TabsTrigger>
          </TabsList>
        </div>

        {/* Ask AI Tab */}
        <TabsContent value="ask-ai">
          <div className="flex flex-col lg:flex-row gap-4 lg:gap-6 h-auto lg:h-[calc(100vh-280px)]">
            <div className="flex-1 min-h-[400px] lg:min-h-0 bg-white rounded-xl border border-border shadow-elevated overflow-hidden">
              <ChatPanel
                messages={messages}
                onSend={handleSendMessage}
                isLoading={askQuestion.isPending}
                onSelectCitation={setSelectedCitations}
                onCitationClick={setChunkPreview}
              />
            </div>
            <div className="w-full lg:w-80 bg-white rounded-xl border border-border shadow-elevated p-5 overflow-y-auto max-h-[400px] lg:max-h-none">
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
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-display font-semibold text-foreground">Risk Analysis</h3>
                  <Button
                    onClick={() => runContractReview.mutate(undefined)}
                    disabled={runContractReview.isPending}
                    size="sm"
                  >
                    {runContractReview.isPending ? (
                      <><Loader2 className="h-4 w-4 animate-spin" /> Analyzing...</>
                    ) : (
                      <><Shield className="h-4 w-4" /> {contractReview.data?.exists ? "Re-run Analysis" : "Run Analysis"}</>
                    )}
                  </Button>
                </div>

                {contractReview.isLoading && (
                  <div className="flex items-center justify-center py-12">
                    <Loader2 className="h-5 w-5 animate-spin text-muted" />
                    <span className="ml-2 text-sm text-muted">Loading review...</span>
                  </div>
                )}

                {!contractReview.isLoading && !contractReview.data?.exists && !runContractReview.isPending && (
                  <div className="text-center py-12 text-muted">
                    <Shield className="h-10 w-10 mx-auto mb-3 opacity-30" />
                    <p className="text-sm">No contract review yet. Click "Run Analysis" to analyze this document for risks.</p>
                  </div>
                )}

                {contractReview.data?.exists && contractReview.data.risks && (
                  <div className="space-y-3">
                    {contractReview.data.risks.map((risk, idx) => (
                      <div
                        key={idx}
                        className={cn(
                          "rounded-xl border p-4 transition-colors hover:shadow-sm",
                          risk.risk_level === "high" ? "border-red-200 bg-red-50/60" :
                          risk.risk_level === "medium" ? "border-amber-200 bg-amber-50/60" :
                          "border-emerald-200 bg-emerald-50/60"
                        )}
                      >
                        <div className="flex items-start justify-between">
                          <div className="flex items-start gap-3">
                            <AlertTriangle className={cn(
                              "h-5 w-5 mt-0.5 shrink-0",
                              risk.risk_level === "high" ? "text-red-600" :
                              risk.risk_level === "medium" ? "text-amber-600" :
                              "text-emerald-600"
                            )} />
                            <div>
                              <p className="font-medium text-foreground text-sm">{risk.clause}</p>
                              <p className="text-sm text-muted mt-1">{risk.explanation}</p>
                              {risk.remedy && (
                                <p className="text-xs text-blue-700 mt-1.5">Remedy: {risk.remedy}</p>
                              )}
                            </div>
                          </div>
                          <Badge variant={risk.risk_level === "high" ? "error" : risk.risk_level === "medium" ? "review" : "active"}>
                            {risk.risk_level.charAt(0).toUpperCase() + risk.risk_level.slice(1)} Risk
                          </Badge>
                        </div>
                      </div>
                    ))}
                    {contractReview.data.risks.length === 0 && (
                      <p className="text-sm text-muted text-center py-6">No risks identified.</p>
                    )}
                  </div>
                )}
              </div>
            </div>

            <div className="space-y-4">
              {contractReview.data?.exists && (
                <>
                  <div className="bg-white rounded-xl border border-border shadow-elevated p-6">
                    <h4 className="font-display font-semibold text-foreground mb-4">Summary</h4>
                    <div className="space-y-3">
                      <div className="flex justify-between text-sm">
                        <span className="text-muted">Total Clauses Analyzed</span>
                        <span className="font-medium">{contractReview.data.summary?.total_clauses ?? 0}</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-muted">High Risk</span>
                        <span className="font-medium text-red-700">{contractReview.data.summary?.high_risk ?? 0}</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-muted">Medium Risk</span>
                        <span className="font-medium text-amber-700">{contractReview.data.summary?.medium_risk ?? 0}</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-muted">Low Risk</span>
                        <span className="font-medium text-emerald-700">{contractReview.data.summary?.low_risk ?? 0}</span>
                      </div>
                      <div className="border-t border-border pt-3">
                        <div className="flex justify-between text-sm mb-2">
                          <span className="text-muted">Overall Score</span>
                          <span className={cn(
                            "font-bold",
                            (contractReview.data.overall_score ?? 0) >= 70 ? "text-emerald-700" :
                            (contractReview.data.overall_score ?? 0) >= 40 ? "text-amber-700" : "text-red-700"
                          )}>{contractReview.data.overall_score ?? 0}/100</span>
                        </div>
                        <Progress value={contractReview.data.overall_score ?? 0} />
                      </div>
                    </div>
                  </div>

                  {contractReview.data.missing_clauses && contractReview.data.missing_clauses.length > 0 && (
                    <div className="bg-white rounded-xl border border-border shadow-elevated p-6">
                      <h4 className="font-display font-semibold text-foreground mb-3">Missing Clauses</h4>
                      <ul className="space-y-2 text-sm">
                        {contractReview.data.missing_clauses.map((clause) => (
                          <li key={clause} className="flex items-center gap-2 text-muted">
                            <div className="h-1.5 w-1.5 rounded-full bg-foreground" />
                            {clause}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        </TabsContent>

        {/* Draft Assistant Tab */}
        <TabsContent value="draft-assistant">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 space-y-4">
              <div className="bg-white rounded-xl border border-border shadow-elevated p-6">
                <h3 className="text-lg font-display font-semibold text-foreground mb-2">Generate New Draft</h3>
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
                        <SelectItem value="Legal Brief">Legal Brief</SelectItem>
                        <SelectItem value="Legal Memorandum">Legal Memorandum</SelectItem>
                        <SelectItem value="Motion">Motion</SelectItem>
                        <SelectItem value="Response Letter">Response Letter</SelectItem>
                        <SelectItem value="Executive Summary">Executive Summary</SelectItem>
                        <SelectItem value="Non-Disclosure Agreement">Non-Disclosure Agreement</SelectItem>
                        <SelectItem value="Employment Contract">Employment Contract</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-foreground mb-2">Instructions</label>
                    <textarea
                      value={draftInstructions}
                      onChange={(e) => setDraftInstructions(e.target.value)}
                      placeholder="Describe what you need drafted. Be specific about the audience, key points to cover, and any constraints..."
                      className="flex min-h-[120px] w-full rounded-lg border border-input bg-white px-3 py-2 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-foreground/10 focus:border-foreground/20 resize-none"
                    />
                  </div>

                  <Button
                    disabled={!draftType || !draftInstructions.trim() || createDraft.isPending}
                    onClick={() => createDraft.mutate({ documentType: draftType, instructions: draftInstructions })}
                  >
                    {createDraft.isPending ? (
                      <><Loader2 className="h-4 w-4 animate-spin" /> Generating...</>
                    ) : (
                      <><PenLine className="h-4 w-4" /> Generate Draft</>
                    )}
                  </Button>
                </div>
              </div>

              {/* Show selected or just-generated draft */}
              {(() => {
                const viewDraft = selectedDraft || createDraft.data
                if (!viewDraft) return null
                return (
                  <div className="bg-white rounded-xl border border-border shadow-elevated p-6">
                    <div className="flex items-center justify-between mb-4">
                      <h4 className="font-display font-semibold text-foreground">
                        {selectedDraft ? "" : "Generated: "}{viewDraft.document_type}
                      </h4>
                      <span className="text-xs text-muted">{formatRelativeTime(viewDraft.created_at)}</span>
                    </div>
                    <div className="prose prose-sm max-w-none text-foreground border border-border rounded-lg p-4 bg-surface/50 max-h-[500px] overflow-y-auto">
                      <ReactMarkdown>{viewDraft.content}</ReactMarkdown>
                    </div>
                    {viewDraft.sources && viewDraft.sources.length > 0 && (
                      <div className="mt-4 border-t border-border pt-3">
                        <p className="text-xs font-medium text-muted mb-2">Sources Referenced</p>
                        <div className="flex flex-wrap gap-2">
                          {viewDraft.sources.map((src, i) => (
                            <Badge key={i} variant="closed" className="text-xs">
                              {src.document_name}{src.page_num && `, p.${src.page_num}`}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )
              })()}
            </div>

            {/* Draft history sidebar */}
            <div className="space-y-4">
              <div className="bg-white rounded-xl border border-border shadow-elevated p-6">
                <h4 className="font-display font-semibold text-foreground mb-4">Draft History</h4>
                {drafts.isLoading && (
                  <div className="flex items-center justify-center py-6">
                    <Loader2 className="h-4 w-4 animate-spin text-muted" />
                  </div>
                )}
                {!drafts.isLoading && (!drafts.data || drafts.data.length === 0) && (
                  <p className="text-sm text-muted text-center py-4">No drafts yet.</p>
                )}
                {drafts.data && drafts.data.length > 0 && (
                  <div className="space-y-3 max-h-[400px] overflow-y-auto">
                    {drafts.data.map((draft) => (
                      <div
                        key={draft.id}
                        className={cn(
                          "rounded-lg border p-3 hover:bg-surface/50 transition-colors cursor-pointer",
                          selectedDraft?.id === draft.id ? "border-foreground/30 bg-surface/50" : "border-border"
                        )}
                        onClick={() => setSelectedDraft(draft)}
                      >
                        <p className="text-sm font-medium text-foreground">{draft.document_type}</p>
                        <p className="text-xs text-muted mt-1 line-clamp-2">{draft.instructions}</p>
                        <p className="text-xs text-muted mt-1">{formatRelativeTime(draft.created_at)}</p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        </TabsContent>

        {/* Audit Log Tab */}
        <TabsContent value="audit-log">
          <div className="bg-white rounded-xl border border-border shadow-elevated">
            <div className="p-4 border-b border-border flex items-center justify-between">
              <h3 className="font-display font-semibold text-foreground">Activity Log</h3>
            </div>
            {auditLog.isLoading && (
              <div className="flex items-center justify-center py-12">
                <Loader2 className="h-5 w-5 animate-spin text-muted" />
                <span className="ml-2 text-sm text-muted">Loading activity log...</span>
              </div>
            )}
            {!auditLog.isLoading && (!auditLog.data || auditLog.data.length === 0) && (
              <div className="text-center py-12 text-muted">
                <ClipboardList className="h-10 w-10 mx-auto mb-3 opacity-30" />
                <p className="text-sm">No activity recorded yet.</p>
              </div>
            )}
            {auditLog.data && auditLog.data.length > 0 && (
              <div className="divide-y divide-border">
                {auditLog.data.map((entry) => (
                  <div key={entry.id} className="px-4 py-3 flex items-start gap-4">
                    <Badge variant={
                      entry.action === "query" ? "default" :
                      entry.action.includes("upload") || entry.action.includes("created") ? "active" :
                      entry.action.includes("delete") || entry.action.includes("cancel") ? "error" :
                      entry.action.includes("review") || entry.action.includes("draft") ? "review" :
                      "default"
                    }>
                      {entry.action.replace(/_/g, " ")}
                    </Badge>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm text-foreground">{entry.details || entry.action}</p>
                      <p className="text-xs text-muted mt-0.5">by {entry.user}</p>
                    </div>
                    <span className="text-xs text-muted shrink-0">{formatRelativeTime(entry.created_at)}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </TabsContent>
      </Tabs>
    </AppLayout>
  )
}
