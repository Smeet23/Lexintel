"use client"

import React from "react"
import { FileText, ExternalLink, Download, BookOpen } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import type { Citation } from "@/lib/types"

interface CitationPanelProps {
  citations: Citation[]
  className?: string
}

export default function CitationPanel({ citations, className }: CitationPanelProps) {
  if (citations.length === 0) {
    return (
      <div className={cn("flex flex-col items-center justify-center py-12 text-center", className)}>
        <div className="h-10 w-10 rounded-lg bg-surface flex items-center justify-center mb-3">
          <BookOpen className="h-5 w-5 text-muted" />
        </div>
        <p className="text-sm font-medium text-foreground">No citations yet</p>
        <p className="text-xs text-muted mt-1">Citations will appear here when you ask questions</p>
      </div>
    )
  }

  return (
    <div className={cn("space-y-3", className)}>
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-semibold text-foreground">
          Supporting Authorities
        </h4>
        <Button variant="ghost" size="sm" className="h-7 text-xs text-muted">
          <Download className="h-3 w-3 mr-1" />
          Export
        </Button>
      </div>

      <div className="space-y-2">
        {citations.map((citation, idx) => (
          <div
            key={idx}
            className="rounded-lg border border-border bg-white p-3 hover:border-accent/30 hover:shadow-sm transition-all cursor-pointer group"
          >
            <div className="flex items-start gap-3">
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-accent/10 text-accent">
                <FileText className="h-4 w-4" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <p className="text-sm font-medium text-foreground truncate">
                    {citation.documentName}
                  </p>
                  <ExternalLink className="h-3 w-3 text-muted opacity-0 group-hover:opacity-100 transition-opacity shrink-0" />
                </div>
                <p className="text-xs text-muted mt-0.5">
                  Page {citation.pageNumber}
                  {citation.section && <> &middot; {citation.section}</>}
                </p>
                {citation.excerpt && (
                  <p className="text-xs text-muted/80 mt-1.5 line-clamp-2 italic">
                    &ldquo;{citation.excerpt}&rdquo;
                  </p>
                )}
              </div>
              <div className="shrink-0">
                <span
                  className={cn(
                    "inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium",
                    citation.relevanceScore >= 0.8
                      ? "bg-emerald-50 text-emerald-700"
                      : citation.relevanceScore >= 0.6
                        ? "bg-amber-50 text-amber-700"
                        : "bg-slate-100 text-slate-600"
                  )}
                >
                  {Math.round(citation.relevanceScore * 100)}%
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="rounded-lg bg-surface border border-border p-3">
        <p className="text-xs text-muted">
          <span className="font-medium text-foreground">{citations.length} sources</span> referenced &middot;
          Average relevance: {Math.round((citations.reduce((a, c) => a + c.relevanceScore, 0) / citations.length) * 100)}%
        </p>
      </div>
    </div>
  )
}
