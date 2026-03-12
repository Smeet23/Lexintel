"use client"

import React from "react"
import { FileText, ExternalLink, Download, BookOpen } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import type { Citation } from "@/lib/types"

interface CitationPanelProps {
  citations: Citation[]
  className?: string
  onCitationClick?: (citation: Citation) => void
}

export default function CitationPanel({ citations, className, onCitationClick }: CitationPanelProps) {
  if (citations.length === 0) {
    return (
      <div className={cn("flex flex-col items-center justify-center py-12 text-center", className)}>
        <div className="h-10 w-10 rounded-xl bg-surface flex items-center justify-center mb-3">
          <BookOpen className="h-4 w-4 text-muted" />
        </div>
        <p className="font-display text-[15px] text-foreground">No citations yet</p>
        <p className="text-[12px] text-muted mt-1">Ask a question to see supporting authorities</p>
      </div>
    )
  }

  return (
    <div className={cn("space-y-3", className)}>
      <div className="flex items-center justify-between">
        <h4 className="text-[12px] font-medium text-muted uppercase tracking-[0.06em]">
          Supporting Authorities
        </h4>
        <Button variant="ghost" size="sm" className="h-6 text-[11px] text-muted">
          <Download className="h-3 w-3 mr-1" />
          Export
        </Button>
      </div>

      <div className="space-y-2">
        {citations.map((citation, idx) => (
          <button
            type="button"
            key={idx}
            onClick={() => onCitationClick?.(citation)}
            className="w-full text-left rounded-xl border border-border bg-white p-3 hover:shadow-elevated hover:border-border-strong transition-all duration-200 cursor-pointer group"
          >
            <div className="flex items-start gap-2.5">
              <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-surface text-muted group-hover:bg-surface-hover transition-colors">
                <FileText className="h-3.5 w-3.5" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-1.5">
                  <p className="text-[13px] font-medium text-foreground truncate">{citation.documentName}</p>
                  <ExternalLink className="h-3 w-3 text-muted opacity-0 group-hover:opacity-100 transition-opacity shrink-0" />
                </div>
                <p className="text-[11px] text-muted mt-0.5">
                  Page {citation.pageNumber}
                  {citation.section && <> &middot; {citation.section}</>}
                </p>
                {citation.excerpt && (
                  <p className="text-[11px] text-muted/70 mt-1.5 line-clamp-2 italic leading-relaxed">
                    &ldquo;{citation.excerpt}&rdquo;
                  </p>
                )}
              </div>
              <span className={cn(
                "shrink-0 font-mono text-[11px] font-medium px-1.5 py-0.5 rounded-md border",
                citation.relevanceScore >= 0.8
                  ? "bg-emerald-50 text-emerald-700 border-emerald-200/60"
                  : citation.relevanceScore >= 0.6
                    ? "bg-amber-50 text-amber-700 border-amber-200/60"
                    : "bg-surface text-muted border-border"
              )}>
                {Math.round(citation.relevanceScore * 100)}%
              </span>
            </div>
          </button>
        ))}
      </div>

      <div className="rounded-xl bg-surface/60 border border-border p-3">
        <p className="text-[11px] text-muted">
          <span className="font-medium text-foreground">{citations.length} sources</span> &middot;
          Avg. relevance: {Math.round((citations.reduce((a, c) => a + c.relevanceScore, 0) / citations.length) * 100)}%
        </p>
      </div>
    </div>
  )
}
