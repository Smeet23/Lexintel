"use client"

import React, { useState, useMemo, useRef } from "react"
import {
  FileText,
  Plus,
  Loader2,
  AlertTriangle,
  Layers,
  ChevronRight,
  Download,
  Trash2,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog"
import { cn } from "@/lib/utils"
import {
  ChunkCard,
  groupChunksByPage,
  uniqueSections,
} from "@/components/DocumentViewer"
import { useMatterDocuments, useMatterChunks, useUploadMatterDocument, useDeleteDocument } from "@/hooks/use-matters"
import { getMatterDocumentDownloadUrl } from "@/lib/api-services"
import type { DocumentResponse } from "@/lib/api-services"

interface DocumentTabProps {
  matterId: string
}

const FILE_TYPE_ICON: Record<string, string> = {
  pdf: "text-red-500 bg-red-50",
  docx: "text-blue-500 bg-blue-50",
  txt: "text-slate-500 bg-slate-50",
}

const STATUS_BADGE: Record<string, "active" | "processing" | "error" | "closed"> = {
  ready: "active",
  processing: "processing",
  error: "error",
  cancelled: "closed",
}

export default function DocumentTab({ matterId }: DocumentTabProps) {
  const { data: documents, isLoading: docsLoading } = useMatterDocuments(matterId)
  const uploadDoc = useUploadMatterDocument(matterId)
  const deleteDoc = useDeleteDocument(matterId)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const [selectedDocId, setSelectedDocId] = useState<string | null>(null)
  const [deleteTarget, setDeleteTarget] = useState<DocumentResponse | null>(null)

  // Auto-select first document when loaded
  const activeDocId = selectedDocId ?? documents?.[0]?.id ?? null
  const activeDoc = documents?.find((d) => d.id === activeDocId) ?? null

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    uploadDoc.mutate({ file })
    e.target.value = "" // reset input
  }

  const confirmDelete = () => {
    if (!deleteTarget) return
    deleteDoc.mutate(deleteTarget.id, {
      onSuccess: () => {
        if (selectedDocId === deleteTarget.id) setSelectedDocId(null)
        setDeleteTarget(null)
      },
    })
  }

  if (docsLoading) {
    return (
      <div className="flex items-center justify-center h-[calc(100vh-280px)]">
        <Loader2 className="h-6 w-6 animate-spin text-muted" />
        <span className="ml-2 text-sm text-muted">Loading documents...</span>
      </div>
    )
  }

  if (!documents || documents.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-[calc(100vh-280px)] gap-4">
        <div className="h-16 w-16 rounded-2xl bg-slate-100 flex items-center justify-center">
          <Layers className="h-8 w-8 text-slate-400" />
        </div>
        <div className="text-center">
          <p className="font-medium text-foreground">No documents yet</p>
          <p className="text-sm text-muted mt-1">Upload a document to get started.</p>
        </div>
        <div>
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.docx,.txt"
            className="hidden"
            onChange={handleFileUpload}
          />
          <Button onClick={() => fileInputRef.current?.click()} disabled={uploadDoc.isPending}>
            {uploadDoc.isPending ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Plus className="h-4 w-4" />
            )}
            Upload Document
          </Button>
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-col md:flex-row gap-0 h-auto md:h-[calc(100vh-280px)] bg-white rounded-xl border border-border shadow-elevated overflow-hidden">
      {/* Left sidebar — document list */}
      <div className="w-full md:w-64 shrink-0 border-b md:border-b-0 md:border-r border-border bg-surface flex flex-col overflow-hidden max-h-[200px] md:max-h-none">
        <div className="px-4 py-3 border-b border-border flex items-center justify-between">
          <p className="text-xs font-semibold text-muted uppercase tracking-wider">
            Documents ({documents.length})
          </p>
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.docx,.txt"
            className="hidden"
            onChange={handleFileUpload}
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={uploadDoc.isPending}
            className="h-6 w-6 inline-flex items-center justify-center rounded-md text-muted hover:text-foreground hover:bg-white transition-colors cursor-pointer"
            title="Add document"
          >
            {uploadDoc.isPending ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <Plus className="h-3.5 w-3.5" />
            )}
          </button>
        </div>

        <div className="flex-1 overflow-y-auto py-1">
          {documents.map((doc) => (
            <div
              key={doc.id}
              className={cn(
                "w-full flex items-start gap-3 px-4 py-3 text-left transition-colors hover:bg-white group cursor-pointer",
                activeDocId === doc.id
                  ? "bg-white border-l-2 border-l-accent"
                  : "border-l-2 border-l-transparent"
              )}
              onClick={() => setSelectedDocId(doc.id)}
            >
              <div
                className={cn(
                  "h-8 w-8 rounded-lg flex items-center justify-center shrink-0 mt-0.5",
                  FILE_TYPE_ICON[doc.file_type] || "text-slate-500 bg-slate-50"
                )}
              >
                <FileText className="h-4 w-4" />
              </div>
              <div className="min-w-0 flex-1">
                <p className={cn(
                  "text-sm truncate",
                  activeDocId === doc.id ? "font-medium text-foreground" : "text-muted"
                )}>
                  {doc.name}
                </p>
                <div className="flex items-center gap-1.5 mt-1 flex-wrap">
                  <Badge variant={STATUS_BADGE[doc.status] || "closed"} className="text-[10px] px-1.5 py-0">
                    {doc.status === "ready" ? "Ready" : doc.status === "processing" ? "Processing" : doc.status}
                  </Badge>
                  {doc.document_type && doc.document_type !== "other" && (
                    <span className="text-[10px] px-1.5 py-0 rounded bg-purple-50 text-purple-700 border border-purple-100">
                      {doc.document_type}
                    </span>
                  )}
                  {doc.jurisdiction && doc.jurisdiction !== "unknown" && (
                    <span className="text-[10px] px-1.5 py-0 rounded bg-emerald-50 text-emerald-700 border border-emerald-100">
                      {doc.jurisdiction}
                    </span>
                  )}
                  {doc.chunk_count > 0 && (
                    <span className="text-[10px] text-muted">{doc.chunk_count} chunks</span>
                  )}
                </div>
              </div>
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  setDeleteTarget(doc)
                }}
                disabled={deleteDoc.isPending}
                className="h-6 w-6 shrink-0 mt-0.5 inline-flex items-center justify-center rounded-md text-muted opacity-0 group-hover:opacity-100 hover:text-red-600 hover:bg-red-50 transition-all cursor-pointer"
                title="Delete document"
              >
                <Trash2 className="h-3.5 w-3.5" />
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Main area — chunk viewer for selected document */}
      {activeDoc ? (
        <DocumentChunkViewer
          matterId={matterId}
          document={activeDoc}
        />
      ) : (
        <div className="flex-1 flex items-center justify-center text-muted text-sm">
          Select a document to view
        </div>
      )}

      {/* Delete confirmation dialog */}
      <Dialog open={!!deleteTarget} onOpenChange={(open) => !open && setDeleteTarget(null)}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <div className="mx-auto mb-3 h-12 w-12 rounded-full bg-red-50 flex items-center justify-center">
              <AlertTriangle className="h-6 w-6 text-red-500" />
            </div>
            <DialogTitle className="text-center">Delete Document</DialogTitle>
            <DialogDescription className="text-center">
              Are you sure you want to delete{" "}
              <span className="font-medium text-foreground">{deleteTarget?.name}</span>?
              This will permanently remove the document and all its chunks.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="sm:justify-center gap-2">
            <Button
              variant="outline"
              onClick={() => setDeleteTarget(null)}
              disabled={deleteDoc.isPending}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={confirmDelete}
              disabled={deleteDoc.isPending}
            >
              {deleteDoc.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin mr-1.5" />
                  Deleting...
                </>
              ) : (
                <>
                  <Trash2 className="h-4 w-4 mr-1.5" />
                  Delete
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}

// ============================================
// Chunk viewer for a single document
// ============================================

interface DocumentChunkViewerProps {
  matterId: string
  document: DocumentResponse
}

function DocumentChunkViewer({ matterId, document: doc }: DocumentChunkViewerProps) {
  const { data: chunks, isLoading, isError } = useMatterChunks(matterId, doc.id)
  const [activeSection, setActiveSection] = useState<string | null>(null)

  const pageGroups = useMemo(() => groupChunksByPage(chunks ?? []), [chunks])
  const sections = useMemo(() => uniqueSections(chunks ?? []), [chunks])

  const downloadUrl = getMatterDocumentDownloadUrl(matterId, doc.id)

  function scrollToSection(sectionName: string) {
    setActiveSection(sectionName)
    const target = chunks?.find((c) => c.section_name === sectionName)
    if (!target) return
    const el = globalThis.document.getElementById(`chunk-${target.id}`)
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "start" })
    }
  }

  if (doc.status === "processing") {
    return (
      <div className="flex-1 flex flex-col items-center justify-center gap-3 p-8">
        <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
        <p className="text-sm font-medium text-foreground">Processing document...</p>
        <p className="text-xs text-muted">Chunks will appear here once processing is complete.</p>
      </div>
    )
  }

  if (isLoading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <Loader2 className="h-6 w-6 animate-spin text-accent mr-2" />
        <span className="text-sm text-muted">Loading chunks...</span>
      </div>
    )
  }

  if (isError) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center gap-2 p-8">
        <AlertTriangle className="h-8 w-8 text-red-500" />
        <p className="text-sm font-medium text-foreground">Failed to load chunks</p>
      </div>
    )
  }

  if (!chunks || chunks.length === 0) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center gap-2 p-8">
        <Layers className="h-8 w-8 text-muted" />
        <p className="text-sm font-medium text-foreground">No chunks available</p>
        <p className="text-xs text-muted">The document may still be processing.</p>
      </div>
    )
  }

  return (
    <div className="flex-1 flex flex-col min-h-0">
      {/* Toolbar */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-border bg-white shrink-0">
        <div className="flex items-center gap-2 min-w-0">
          <FileText className="h-4 w-4 text-muted shrink-0" />
          <p className="text-sm font-medium text-foreground truncate">{doc.name}</p>
          <span className="text-xs text-muted shrink-0">
            {chunks.length} chunks &middot; {pageGroups.length} pages
          </span>
        </div>
        <a
          href={downloadUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1.5 text-xs text-muted hover:text-foreground transition-colors"
        >
          <Download className="h-3.5 w-3.5" />
          {doc.file_type === "pdf" ? "View Original" : "Download"}
        </a>
      </div>

      {/* Document summary bar (if available) */}
      {doc.summary && (
        <div className="px-4 py-2 border-b border-border bg-blue-50/50 shrink-0">
          <p className="text-xs text-blue-800 leading-relaxed">{doc.summary}</p>
        </div>
      )}

      {/* Content area: section sidebar + chunks */}
      <div className="flex flex-1 min-h-0">
        {/* Section sidebar */}
        {sections.length > 0 && (
          <div className="w-56 shrink-0 border-r border-border bg-surface/50 flex flex-col overflow-hidden">
            <div className="px-3 py-2 border-b border-border">
              <p className="text-[10px] font-semibold text-muted uppercase tracking-wider">
                Sections ({sections.length})
              </p>
            </div>
            <div className="flex-1 overflow-y-auto py-1">
              {sections.map((section) => (
                <button
                  key={section}
                  onClick={() => scrollToSection(section)}
                  className={cn(
                    "w-full flex items-center gap-1.5 px-3 py-1.5 text-left text-xs transition-colors hover:bg-white hover:text-foreground group",
                    activeSection === section
                      ? "text-accent font-medium bg-white"
                      : "text-muted"
                  )}
                >
                  <ChevronRight
                    className={cn(
                      "h-2.5 w-2.5 shrink-0 transition-transform",
                      activeSection === section ? "text-accent" : "text-transparent group-hover:text-muted"
                    )}
                  />
                  <span className="truncate">{section}</span>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Chunks scroll area */}
        <div className="flex-1 overflow-y-auto">
          {pageGroups.map(({ pageNum, chunks: pageChunks }) => (
            <div key={pageNum} className="border-b border-border last:border-b-0">
              <div className="sticky top-0 z-10 px-6 py-2 bg-surface/95 backdrop-blur-sm border-b border-border flex items-center gap-2">
                <FileText className="h-3.5 w-3.5 text-muted" />
                <span className="text-xs font-semibold text-muted uppercase tracking-wider">
                  Page {pageNum}
                </span>
                <span className="text-xs text-muted/60">
                  &middot; {pageChunks.length} chunk{pageChunks.length !== 1 ? "s" : ""}
                </span>
              </div>
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
    </div>
  )
}
