"use client"

import React from "react"
import { FileText, ExternalLink, Download, BookOpen, CheckCircle2, AlertTriangle, XCircle, HelpCircle, ShieldCheck, ShieldAlert, ShieldX, Scale, Clock } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn, isSafeUrl } from "@/lib/utils"
import type { Citation, VerificationStatus } from "@/lib/types"

interface CitationPanelProps {
  citations: Citation[]
  className?: string
  onCitationClick?: (citation: Citation) => void
}

const VERIFICATION_CONFIG: Record<VerificationStatus, {
  bg: string; text: string; border: string; label: string; Icon: React.ElementType
}> = {
  verified: { bg: "bg-emerald-50", text: "text-emerald-700", border: "border-emerald-200/60", label: "Verified", Icon: CheckCircle2 },
  partial: { bg: "bg-amber-50", text: "text-amber-700", border: "border-amber-200/60", label: "Partial", Icon: AlertTriangle },
  unverified: { bg: "bg-red-50", text: "text-red-700", border: "border-red-200/60", label: "Unverified", Icon: XCircle },
  not_found: { bg: "bg-red-100", text: "text-red-800", border: "border-red-300/60", label: "Not Found", Icon: XCircle },
  unverifiable: { bg: "bg-gray-50", text: "text-gray-600", border: "border-gray-200/60", label: "N/A", Icon: HelpCircle },
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

  // Count verified citations
  const verifiedCount = citations.filter(c => c.verification?.status === "verified").length
  const totalVerifiable = citations.filter(c => c.verification).length

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
        {citations.map((citation, idx) => {
          const v = citation.verification
          const vConfig = v ? (VERIFICATION_CONFIG[v.status] ?? VERIFICATION_CONFIG.unverifiable) : null

          return (
            <button
              type="button"
              key={idx}
              onClick={() => onCitationClick?.(citation)}
              className="w-full text-left rounded-xl border border-border bg-white p-3 hover:shadow-elevated hover:border-border-strong transition-all duration-200 cursor-pointer group"
            >
              <div className="flex items-start gap-2.5">
                {/* Source icon with index badge */}
                <div className="relative">
                  <div className={cn(
                    "flex h-7 w-7 shrink-0 items-center justify-center rounded-lg transition-colors",
                    citation.sourceType === "case_law"
                      ? "bg-blue-50 text-blue-600 group-hover:bg-blue-100"
                      : "bg-surface text-muted group-hover:bg-surface-hover"
                  )}>
                    <FileText className="h-3.5 w-3.5" />
                  </div>
                  {/* Index badge */}
                  {v && (
                    <span className={cn(
                      "absolute -top-1.5 -right-1.5 flex h-4 w-4 items-center justify-center rounded-full text-[9px] font-bold border",
                      vConfig?.bg, vConfig?.text, vConfig?.border,
                    )}>
                      {v.index}
                    </span>
                  )}
                </div>

                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-1.5">
                    <p className="text-[13px] font-medium text-foreground truncate">
                      {v?.caseName || citation.documentName}
                    </p>
                    {citation.sourceType === "case_law" && citation.url && isSafeUrl(citation.url) ? (
                      <a
                        href={citation.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        onClick={(e) => e.stopPropagation()}
                        className="shrink-0"
                      >
                        <ExternalLink className="h-3 w-3 text-blue-500 hover:text-blue-700 transition-colors" />
                      </a>
                    ) : v?.sourceUrl && isSafeUrl(v.sourceUrl) ? (
                      <a
                        href={v.sourceUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        onClick={(e) => e.stopPropagation()}
                        className="shrink-0"
                      >
                        <ExternalLink className="h-3 w-3 text-blue-500 hover:text-blue-700 transition-colors" />
                      </a>
                    ) : (
                      <ExternalLink className="h-3 w-3 text-muted opacity-0 group-hover:opacity-100 transition-opacity shrink-0" />
                    )}
                  </div>

                  {/* Metadata row */}
                  <div className="flex items-center gap-1.5 mt-0.5 flex-wrap">
                    {citation.sourceType === "case_law" ? (
                      <span className="inline-flex items-center rounded-md bg-blue-50 px-1.5 py-0.5 text-[10px] font-medium text-blue-700 border border-blue-200/60">
                        Case Law
                      </span>
                    ) : (
                      <span className="text-[11px] text-muted">
                        Page {citation.pageNumber}
                      </span>
                    )}
                    {citation.section && <span className="text-[11px] text-muted">&middot; {citation.section}</span>}

                    {/* Verification badge */}
                    {v && vConfig && (
                      <span className={cn(
                        "inline-flex items-center gap-0.5 rounded-md px-1.5 py-0.5 text-[10px] font-medium border",
                        vConfig.bg, vConfig.text, vConfig.border,
                      )}>
                        <vConfig.Icon className="h-2.5 w-2.5" />
                        {vConfig.label}
                      </span>
                    )}

                    {/* Case status */}
                    {v?.caseStatus && v.caseStatus !== "unknown" && (
                      <span className={cn(
                        "inline-flex items-center gap-0.5 rounded-md px-1.5 py-0.5 text-[10px] font-medium",
                        v.caseStatus === "good_law" ? "bg-emerald-50 text-emerald-700" :
                          v.caseStatus === "caution" ? "bg-amber-50 text-amber-700" :
                            "bg-red-50 text-red-700"
                      )}>
                        {v.caseStatus === "good_law" && <ShieldCheck className="h-2.5 w-2.5" />}
                        {v.caseStatus === "caution" && <ShieldAlert className="h-2.5 w-2.5" />}
                        {v.caseStatus === "bad_law" && <ShieldX className="h-2.5 w-2.5" />}
                        {v.caseStatus === "good_law" ? "Good Law" :
                          v.caseStatus === "caution" ? "Caution" : "Overruled"}
                      </span>
                    )}
                  </div>

                  {/* Authority badge */}
                  {citation.courtLevel && citation.courtLevel !== "unknown" && (
                    <div className="flex items-center gap-1 mt-0.5">
                      <span className={cn(
                        "inline-flex items-center gap-0.5 rounded-md px-1.5 py-0.5 text-[10px] font-medium border",
                        citation.courtLevel === "supreme" ? "bg-purple-50 text-purple-700 border-purple-200/60" :
                          citation.courtLevel === "appellate" ? "bg-indigo-50 text-indigo-700 border-indigo-200/60" :
                            "bg-slate-50 text-slate-600 border-slate-200/60"
                      )}>
                        <Scale className="h-2.5 w-2.5" />
                        {citation.courtLevel === "supreme" ? "Supreme" :
                          citation.courtLevel === "appellate" ? "Appellate" :
                            citation.courtLevel === "trial" ? "Trial" : citation.courtLevel}
                      </span>
                      {citation.bindingAuthority && (
                        <span className="inline-flex items-center rounded-md bg-purple-50 px-1.5 py-0.5 text-[10px] font-medium text-purple-700 border border-purple-200/60">
                          Binding
                        </span>
                      )}
                    </div>
                  )}

                  {/* Temporal status */}
                  {citation.documentStatus && citation.documentStatus !== "unknown" && citation.documentStatus !== "current" && (
                    <span className={cn(
                      "inline-flex items-center gap-0.5 rounded-md px-1.5 py-0.5 text-[10px] font-medium border mt-0.5",
                      citation.documentStatus === "superseded" ? "bg-orange-50 text-orange-700 border-orange-200/60" :
                        citation.documentStatus === "repealed" ? "bg-red-50 text-red-800 border-red-200/60" :
                          "bg-gray-50 text-gray-600 border-gray-200/60"
                    )}>
                      <Clock className="h-2.5 w-2.5" />
                      {citation.documentStatus === "superseded" ? "Superseded" :
                        citation.documentStatus === "repealed" ? "Repealed" : citation.documentStatus}
                    </span>
                  )}
                  {citation.effectiveDate && (
                    <span className="text-[10px] text-muted-foreground mt-0.5">
                      Effective: {new Date(citation.effectiveDate).toLocaleDateString()}
                    </span>
                  )}

                  {/* Court + date for verified citations */}
                  {v?.court && (
                    <p className="text-[10px] text-muted-foreground mt-0.5">
                      {[v.court, v.date, v.jurisdiction].filter(Boolean).join(" · ")}
                    </p>
                  )}

                  {/* Citation count */}
                  {v?.citationCount !== undefined && v.citationCount > 0 && (
                    <p className="text-[10px] text-muted-foreground mt-0.5">
                      Cited by {v.citationCount.toLocaleString()} cases
                    </p>
                  )}

                  {citation.excerpt && (
                    <p className="text-[11px] text-muted/70 mt-1.5 line-clamp-2 italic leading-relaxed">
                      &ldquo;{citation.excerpt}&rdquo;
                    </p>
                  )}
                </div>

                {/* Confidence score */}
                <span className={cn(
                  "shrink-0 font-mono text-[11px] font-medium px-1.5 py-0.5 rounded-md border",
                  (v?.confidence ?? citation.relevanceScore) >= 0.8
                    ? "bg-emerald-50 text-emerald-700 border-emerald-200/60"
                    : (v?.confidence ?? citation.relevanceScore) >= 0.6
                      ? "bg-amber-50 text-amber-700 border-amber-200/60"
                      : (v?.confidence ?? citation.relevanceScore) >= 0.4
                        ? "bg-red-50 text-red-700 border-red-200/60"
                        : "bg-surface text-muted border-border"
                )}>
                  {Math.round((v?.confidence ?? citation.relevanceScore) * 100)}%
                </span>
              </div>
            </button>
          )
        })}
      </div>

      {/* Summary footer */}
      <div className="rounded-xl bg-surface/60 border border-border p-3">
        <p className="text-[11px] text-muted">
          <span className="font-medium text-foreground">{citations.length} sources</span>
          {totalVerifiable > 0 && (
            <>
              {" "}&middot;{" "}
              <span className={cn(
                "font-medium",
                verifiedCount === totalVerifiable ? "text-emerald-700" : "text-foreground"
              )}>
                {verifiedCount}/{totalVerifiable} verified
              </span>
            </>
          )}
          {" "}&middot;{" "}
          Avg. relevance: {Math.round((citations.reduce((a, c) => a + c.relevanceScore, 0) / citations.length) * 100)}%
        </p>
      </div>
    </div>
  )
}
