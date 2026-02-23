"use client"

import React, { useState, useRef, useEffect, useMemo } from "react"
import * as DialogPrimitive from "@radix-ui/react-dialog"
import { X, Download, FileText, Layers, ChevronRight, Loader2, AlertTriangle, Hash } from "lucide-react"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { getDocumentUrl } from "@/lib/api"
import { useCaseChunks } from "@/hooks/use-cases"
import type { ChunkResponse } from "@/lib/types"

// ============================================
// Section type badge colour mapping
// ============================================

const SECTION_TYPE_VARIANT: Record<string, "default" | "active" | "review" | "closed" | "error" | "processing" | "indexed"> = {
  article: "default",
  section: "default",
  contract_header: "processing",
  exhibit: "review",
  litigation_count: "error",
  litigation_relief: "error",
  statement_of_facts: "active",
  legal_argument: "active",
  clause: "review",
  part: "default",
  regulation: "processing",
  chapter: "default",
  division: "default",
  subdivision: "default",
  title: "processing",
  subtitle: "default",
  numbered_header: "default",
}

function sectionVariant(type: string) {
  return SECTION_TYPE_VARIANT[type] ?? "closed"
}

function humanizeSectionType(type: string): string {
  return type
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ")
}

// ============================================
// Chunk grouping helpers
// ============================================

interface PageGroup {
  pageNum: string
  chunks: ChunkResponse[]
}

function groupChunksByPage(chunks: ChunkResponse[]): PageGroup[] {
  const map = new Map<string, ChunkResponse[]>()
  for (const chunk of chunks) {
    const key = chunk.page_num ?? "unknown"
    const group = map.get(key)
    if (group) {
      group.push(chunk)
    } else {
      map.set(key, [chunk])
    }
  }
  // Sort pages numerically where possible
  const sorted = Array.from(map.entries()).sort(([a], [b]) => {
    const na = parseInt(a, 10)
    const nb = parseInt(b, 10)
    if (!isNaN(na) && !isNaN(nb)) return na - nb
    return a.localeCompare(b)
  })
  return sorted.map(([pageNum, pageChunks]) => ({
    pageNum,
    chunks: pageChunks.sort((a, b) => a.chunk_sequence - b.chunk_sequence),
  }))
}

function uniqueSections(chunks: ChunkResponse[]): string[] {
  const seen = new Set<string>()
  const result: string[] = []
  for (const chunk of chunks) {
    if (chunk.section_name && !seen.has(chunk.section_name)) {
      seen.add(chunk.section_name)
      result.push(chunk.section_name)
    }
  }
  return result
}

// ============================================
// Individual chunk card
// ============================================

interface ChunkCardProps {
  chunk: ChunkResponse
  isHighlighted: boolean
}

function ChunkCard({ chunk, isHighlighted }: ChunkCardProps) {
  return (
    <div
      id={`chunk-${chunk.id}`}
      className={cn(
        "rounded-lg border p-4 transition-all",
        isHighlighted
          ? "border-accent/50 bg-accent/5 shadow-sm"
          : "border-border bg-white hover:border-accent/20 hover:shadow-sm"
      )}
    >
      {/* Header row */}
      <div className="flex items-start justify-between gap-3 mb-2">
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-sm font-semibold text-foreground truncate">
            {chunk.section_name || "Untitled Section"}
          </span>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <Badge variant={sectionVariant(chunk.section_type)}>
            {humanizeSectionType(chunk.section_type)}
          </Badge>
          <span className="inline-flex items-center gap-1 text-xs text-muted bg-surface rounded-full px-2 py-0.5 border border-border">
            <Hash className="h-2.5 w-2.5" />
            {chunk.chunk_sequence}
          </span>
        </div>
      </div>

      {/* Content */}
      <p className="text-sm text-muted leading-relaxed whitespace-pre-wrap">
        {chunk.content}
      </p>
    </div>
  )
}

// ============================================
// Chunks tab — full view with sidebar
// ============================================

interface ChunksViewProps {
  caseId: string
}

function ChunksView({ caseId }: ChunksViewProps) {
  const { data: chunks, isLoading, isError } = useCaseChunks(caseId)
  const [activeSection, setActiveSection] = useState<string | null>(null)

  const pageGroups = useMemo(() => groupChunksByPage(chunks ?? []), [chunks])
  const sections = useMemo(() => uniqueSections(chunks ?? []), [chunks])

  function scrollToSection(sectionName: string) {
    setActiveSection(sectionName)
    // Find the first chunk with that section name
    const target = chunks?.find((c) => c.section_name === sectionName)
    if (!target) return
    const el = document.getElementById(`chunk-${target.id}`)
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "start" })
    }
  }

  if (isLoading) {
    return (
      <div className="flex flex-1 items-center justify-center">
        <Loader2 className="h-6 w-6 animate-spin text-accent mr-2" />
        <span className="text-sm text-muted">Loading chunks...</span>
      </div>
    )
  }

  if (isError) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-2 text-center p-8">
        <AlertTriangle className="h-8 w-8 text-red-500" />
        <p className="text-sm font-medium text-foreground">Failed to load document chunks</p>
        <p className="text-xs text-muted">The chunks endpoint may not be available yet.</p>
      </div>
    )
  }

  if (!chunks || chunks.length === 0) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-2 text-center p-8">
        <Layers className="h-8 w-8 text-muted" />
        <p className="text-sm font-medium text-foreground">No chunks available</p>
        <p className="text-xs text-muted">The document may still be processing.</p>
      </div>
    )
  }

  return (
    <div className="flex flex-1 min-h-0">
      {/* Left sidebar — section navigator */}
      <div className="w-64 shrink-0 border-r border-border bg-surface flex flex-col overflow-hidden">
        <div className="px-4 py-3 border-b border-border">
          <p className="text-xs font-semibold text-muted uppercase tracking-wider">
            Sections ({sections.length})
          </p>
        </div>
        <div className="flex-1 overflow-y-auto py-2">
          {sections.map((section) => (
            <button
              key={section}
              onClick={() => scrollToSection(section)}
              className={cn(
                "w-full flex items-center gap-2 px-4 py-2 text-left text-sm transition-colors hover:bg-white hover:text-foreground group",
                activeSection === section
                  ? "text-accent font-medium bg-white"
                  : "text-muted"
              )}
            >
              <ChevronRight
                className={cn(
                  "h-3 w-3 shrink-0 transition-transform",
                  activeSection === section ? "text-accent" : "text-transparent group-hover:text-muted"
                )}
              />
              <span className="truncate">{section}</span>
            </button>
          ))}
        </div>
        {/* Footer stats */}
        <div className="px-4 py-3 border-t border-border bg-white">
          <p className="text-xs text-muted">
            <span className="font-medium text-foreground">{chunks.length}</span> chunks across{" "}
            <span className="font-medium text-foreground">{pageGroups.length}</span> pages
          </p>
        </div>
      </div>

      {/* Main scroll area */}
      <div className="flex-1 overflow-y-auto">
        {pageGroups.map(({ pageNum, chunks: pageChunks }) => (
          <div key={pageNum} className="border-b border-border last:border-b-0">
            {/* Page header */}
            <div className="sticky top-0 z-10 px-6 py-2 bg-surface/95 backdrop-blur-sm border-b border-border flex items-center gap-2">
              <FileText className="h-3.5 w-3.5 text-muted" />
              <span className="text-xs font-semibold text-muted uppercase tracking-wider">
                Page {pageNum}
              </span>
              <span className="text-xs text-muted/60">
                &middot; {pageChunks.length} chunk{pageChunks.length !== 1 ? "s" : ""}
              </span>
            </div>

            {/* Chunks for this page */}
            <div className="px-6 py-4 space-y-3">
              {pageChunks.map((chunk) => (
                <ChunkCard
                  key={chunk.id}
                  chunk={chunk}
                  isHighlighted={activeSection === chunk.section_name}
                />
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

// ============================================
// Original document tab — iframe / download
// ============================================

interface OriginalViewProps {
  caseId: string
  caseName: string
  isPdf: boolean
}

function OriginalView({ caseId, caseName, isPdf }: OriginalViewProps) {
  const documentUrl = getDocumentUrl(caseId)

  if (isPdf) {
    return (
      <div className="flex flex-1 flex-col min-h-0">
        {/* Toolbar */}
        <div className="flex items-center justify-between px-4 py-2 border-b border-border bg-white shrink-0">
          <p className="text-sm font-medium text-foreground truncate">{caseName}</p>
          <a
            href={documentUrl}
            download
            className="inline-flex items-center gap-1.5 text-xs text-muted hover:text-foreground transition-colors"
          >
            <Download className="h-3.5 w-3.5" />
            Download
          </a>
        </div>
        {/* iframe */}
        <iframe
          src={documentUrl}
          title={caseName}
          className="flex-1 w-full border-0"
          style={{ minHeight: 0 }}
        />
      </div>
    )
  }

  // Non-PDF fallback
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-4 p-8 text-center">
      <div className="h-16 w-16 rounded-2xl bg-slate-100 flex items-center justify-center">
        <FileText className="h-8 w-8 text-slate-500" />
      </div>
      <div>
        <p className="font-semibold text-foreground">{caseName}</p>
        <p className="text-sm text-muted mt-1">
          Inline preview is not available for this file type.
        </p>
      </div>
      <a href={documentUrl} download>
        <Button size="sm">
          <Download className="h-4 w-4" />
          Download Document
        </Button>
      </a>
    </div>
  )
}

// ============================================
// Main DocumentViewer component
// ============================================

interface DocumentViewerProps {
  open: boolean
  onClose: () => void
  caseId: string
  caseName: string
  fileType?: string
}

export default function DocumentViewer({
  open,
  onClose,
  caseId,
  caseName,
  fileType,
}: DocumentViewerProps) {
  const isPdf = !fileType || fileType.toLowerCase() === "pdf"
  const [activeTab, setActiveTab] = useState("original")

  // Reset tab when dialog reopens
  useEffect(() => {
    if (open) setActiveTab("original")
  }, [open])

  return (
    <DialogPrimitive.Root open={open} onOpenChange={(o) => { if (!o) onClose() }}>
      <DialogPrimitive.Portal>
        {/* Overlay */}
        <DialogPrimitive.Overlay className="fixed inset-0 z-50 bg-black/50 backdrop-blur-sm animate-fade-in" />

        {/* Full-screen dialog panel */}
        <DialogPrimitive.Content
          className={cn(
            "fixed inset-4 z-50 flex flex-col rounded-xl border border-border bg-card shadow-2xl animate-slide-up overflow-hidden",
            "md:inset-6 lg:inset-8"
          )}
          aria-describedby={undefined}
        >
          {/* Dialog header */}
          <div className="flex items-center justify-between px-6 py-4 border-b border-border bg-white shrink-0">
            <div className="flex items-center gap-3 min-w-0">
              <div className="h-8 w-8 rounded-lg bg-red-50 flex items-center justify-center shrink-0">
                <FileText className="h-4 w-4 text-red-500" />
              </div>
              <div className="min-w-0">
                <DialogPrimitive.Title className="text-base font-semibold text-foreground truncate">
                  {caseName}
                </DialogPrimitive.Title>
                <p className="text-xs text-muted">Document Viewer</p>
              </div>
            </div>

            {/* Close button */}
            <DialogPrimitive.Close
              onClick={onClose}
              className="ml-4 shrink-0 h-8 w-8 inline-flex items-center justify-center rounded-lg text-muted hover:text-foreground hover:bg-surface transition-colors cursor-pointer"
              aria-label="Close document viewer"
            >
              <X className="h-4 w-4" />
            </DialogPrimitive.Close>
          </div>

          {/* Tabs row */}
          <div className="px-6 bg-white border-b border-border shrink-0">
            <Tabs value={activeTab} onValueChange={setActiveTab}>
              <TabsList className="border-b-0">
                <TabsTrigger value="original">
                  <FileText className="h-4 w-4 mr-2" />
                  Original
                </TabsTrigger>
                <TabsTrigger value="chunks">
                  <Layers className="h-4 w-4 mr-2" />
                  Chunks
                </TabsTrigger>
              </TabsList>
            </Tabs>
          </div>

          {/* Tab content — fills remaining space */}
          <div className="flex flex-1 min-h-0">
            {activeTab === "original" && (
              <OriginalView caseId={caseId} caseName={caseName} isPdf={isPdf} />
            )}
            {activeTab === "chunks" && (
              <ChunksView caseId={caseId} />
            )}
          </div>
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  )
}
