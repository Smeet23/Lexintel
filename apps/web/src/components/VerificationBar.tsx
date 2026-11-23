"use client"

import React, { useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { ChevronDown, ChevronUp, CheckCircle2, AlertTriangle, XCircle, HelpCircle, ShieldCheck, ShieldAlert, ShieldX } from "lucide-react"
import { cn } from "@/lib/utils"
import type { CitationVerification, VerificationSummary, ClaimVerification, ClaimVerificationSummary } from "@/lib/types"

interface VerificationBarProps {
  summary: VerificationSummary
  citations: CitationVerification[]
  claimSummary?: ClaimVerificationSummary
  claims?: ClaimVerification[]
  className?: string
}

const STATUS_COLORS = {
  verified: "bg-emerald-500",
  partial: "bg-amber-500",
  unverified: "bg-red-500",
  not_found: "bg-red-700",
  unverifiable: "bg-gray-400",
}

const STATUS_ICONS = {
  verified: CheckCircle2,
  partial: AlertTriangle,
  unverified: XCircle,
  not_found: XCircle,
  unverifiable: HelpCircle,
}

const STATUS_LABELS = {
  verified: "Verified",
  partial: "Partial",
  unverified: "Unverified",
  not_found: "Not Found",
  unverifiable: "Unverifiable",
}

const STATUS_TEXT_COLORS = {
  verified: "text-emerald-700",
  partial: "text-amber-700",
  unverified: "text-red-700",
  not_found: "text-red-800",
  unverifiable: "text-gray-600",
}

const CASE_STATUS_ICONS = {
  good_law: ShieldCheck,
  caution: ShieldAlert,
  bad_law: ShieldX,
  unknown: HelpCircle,
}

const CLAIM_COLORS = {
  supported: "bg-emerald-500",
  partially_supported: "bg-amber-500",
  unsupported: "bg-red-500",
  uncited: "bg-gray-400",
}

const CLAIM_LABELS = {
  supported: "Grounded",
  partially_supported: "Partial",
  unsupported: "Mismatched",
  uncited: "No Citation",
}

const CLAIM_TEXT_COLORS = {
  supported: "text-emerald-700",
  partially_supported: "text-amber-700",
  unsupported: "text-red-700",
  uncited: "text-gray-600",
}

const ISSUE_LABELS: Record<string, string> = {
  no_citation: "No citation",
  source_not_found: "Source not found",
  contradiction: "Contradiction detected",
  weak_source: "Low relevance source",
  source_conflict: "Sources conflict",
  disagreement: "Models disagree",
  potential_omission: "May omit qualifications",
  conditional: "Conditional claim",
  double_negation: "Double negation",
  has_quantifier: "Quantifier (all/some/none)",
}

export default function VerificationBar({ summary, citations, claimSummary, claims, className }: VerificationBarProps) {
  const [expanded, setExpanded] = useState(false)
  const [claimExpanded, setClaimExpanded] = useState(false)

  if (summary.total === 0 && (!claimSummary || claimSummary.total === 0)) return null

  const segments = [
    { key: "verified", count: summary.verified, color: STATUS_COLORS.verified },
    { key: "partial", count: summary.partial, color: STATUS_COLORS.partial },
    { key: "unverified", count: summary.unverified, color: STATUS_COLORS.unverified },
    { key: "not_found", count: summary.not_found, color: STATUS_COLORS.not_found },
    { key: "unverifiable", count: summary.unverifiable ?? 0, color: STATUS_COLORS.unverifiable },
  ].filter(s => s.count > 0)

  // Summary text
  const verifiedCount = summary.verified
  const totalCount = summary.total
  const allVerified = verifiedCount === totalCount

  return (
    <div className={cn("rounded-lg border border-border bg-surface/50 overflow-hidden", className)}>
      {/* Compact bar */}
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-3 px-3 py-2 hover:bg-surface transition-colors text-left"
      >
        {/* Progress bar */}
        <div className="flex h-1.5 flex-1 rounded-full overflow-hidden bg-gray-100">
          {segments.map(seg => (
            <div
              key={seg.key}
              className={cn("transition-all duration-500", seg.color)}
              style={{ width: `${(seg.count / totalCount) * 100}%` }}
            />
          ))}
        </div>

        {/* Summary text */}
        <span className={cn(
          "text-[11px] font-medium whitespace-nowrap",
          allVerified ? "text-emerald-700" : "text-muted-foreground"
        )}>
          {allVerified ? (
            <span className="flex items-center gap-1">
              <CheckCircle2 className="h-3 w-3" />
              All {totalCount} verified
            </span>
          ) : (
            `${verifiedCount}/${totalCount} verified`
          )}
        </span>

        {/* Expand toggle */}
        {expanded
          ? <ChevronUp className="h-3 w-3 text-muted-foreground shrink-0" />
          : <ChevronDown className="h-3 w-3 text-muted-foreground shrink-0" />
        }
      </button>

      {/* Expanded citation list */}
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="border-t border-border px-3 py-2 space-y-1.5">
              {citations.map((cite) => {
                const StatusIcon = STATUS_ICONS[cite.status] ?? HelpCircle
                const CaseIcon = cite.caseStatus ? CASE_STATUS_ICONS[cite.caseStatus] ?? HelpCircle : null

                return (
                  <div
                    key={cite.index}
                    className="flex items-center gap-2 rounded-md px-2 py-1.5 bg-white border border-border/50 text-[11px]"
                  >
                    {/* Index badge */}
                    <span className={cn(
                      "flex h-5 w-5 items-center justify-center rounded text-[10px] font-bold border shrink-0",
                      cite.status === "verified" ? "bg-emerald-100 text-emerald-700 border-emerald-300" :
                        cite.status === "partial" ? "bg-amber-100 text-amber-700 border-amber-300" :
                          cite.status === "not_found" ? "bg-red-200 text-red-800 border-red-400" :
                            cite.status === "unverified" ? "bg-red-100 text-red-700 border-red-300" :
                              "bg-gray-100 text-gray-600 border-gray-300"
                    )}>
                      {cite.index}
                    </span>

                    {/* Case info */}
                    <div className="flex-1 min-w-0">
                      <p className="font-medium text-foreground truncate">
                        {cite.caseName || cite.rawText}
                      </p>
                      {cite.court && (
                        <p className="text-[10px] text-muted-foreground truncate">
                          {[cite.court, cite.date, cite.jurisdiction].filter(Boolean).join(" · ")}
                        </p>
                      )}
                    </div>

                    {/* Status badges */}
                    <div className="flex items-center gap-1.5 shrink-0">
                      <StatusIcon className={cn("h-3.5 w-3.5", STATUS_TEXT_COLORS[cite.status])} />
                      <span className={cn("text-[10px] font-medium", STATUS_TEXT_COLORS[cite.status])}>
                        {STATUS_LABELS[cite.status]}
                      </span>
                    </div>

                    {/* Case status icon */}
                    {CaseIcon && cite.caseStatus !== "unknown" && (
                      <CaseIcon className={cn(
                        "h-3.5 w-3.5 shrink-0",
                        cite.caseStatus === "good_law" ? "text-emerald-600" :
                          cite.caseStatus === "caution" ? "text-amber-600" :
                            "text-red-600"
                      )} />
                    )}

                    {/* Confidence */}
                    <span className={cn(
                      "text-[10px] font-mono font-medium shrink-0",
                      cite.confidence >= 0.8 ? "text-emerald-700" :
                        cite.confidence >= 0.5 ? "text-amber-700" : "text-red-700"
                    )}>
                      {Math.round(cite.confidence * 100)}%
                    </span>
                  </div>
                )
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Claim Verification Section ─────────────────── */}
      {claimSummary && claimSummary.total > 0 && (
        <>
          <button
            type="button"
            onClick={() => setClaimExpanded(!claimExpanded)}
            className="w-full flex items-center gap-3 px-3 py-2 hover:bg-surface transition-colors text-left border-t border-border"
          >
            {/* Claims progress bar */}
            <div className="flex h-1.5 flex-1 rounded-full overflow-hidden bg-gray-100">
              {[
                { key: "supported", count: claimSummary.supported, color: CLAIM_COLORS.supported },
                { key: "partially_supported", count: claimSummary.partially_supported, color: CLAIM_COLORS.partially_supported },
                { key: "unsupported", count: claimSummary.unsupported, color: CLAIM_COLORS.unsupported },
                { key: "uncited", count: claimSummary.uncited, color: CLAIM_COLORS.uncited },
              ].filter(s => s.count > 0).map(seg => (
                <div
                  key={seg.key}
                  className={cn("transition-all duration-500", seg.color)}
                  style={{ width: `${(seg.count / claimSummary.total) * 100}%` }}
                />
              ))}
            </div>

            <span className={cn(
              "text-[11px] font-medium whitespace-nowrap",
              claimSummary.supported === claimSummary.total ? "text-emerald-700" : "text-muted-foreground"
            )}>
              {claimSummary.supported === claimSummary.total ? (
                <span className="flex items-center gap-1">
                  <CheckCircle2 className="h-3 w-3" />
                  All claims grounded
                </span>
              ) : (
                `${claimSummary.supported}/${claimSummary.total} grounded`
              )}
            </span>

            {claimExpanded
              ? <ChevronUp className="h-3 w-3 text-muted-foreground shrink-0" />
              : <ChevronDown className="h-3 w-3 text-muted-foreground shrink-0" />
            }
          </button>

          <AnimatePresence>
            {claimExpanded && claims && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="overflow-hidden"
              >
                <div className="border-t border-border px-3 py-2 space-y-1.5">
                  {claims.map((claim, idx) => (
                    <div
                      key={idx}
                      className="rounded-md px-2 py-1.5 bg-white border border-border/50 text-[11px]"
                    >
                      <div className="flex items-center gap-2">
                        {/* Status icon */}
                        <span className={cn(
                          "flex h-5 w-5 items-center justify-center rounded shrink-0 border text-[10px] font-bold",
                          claim.status === "supported" ? "bg-emerald-100 text-emerald-700 border-emerald-300" :
                            claim.status === "partially_supported" ? "bg-amber-100 text-amber-700 border-amber-300" :
                              claim.status === "unsupported" ? "bg-red-100 text-red-700 border-red-300" :
                                "bg-gray-100 text-gray-600 border-gray-300"
                        )}>
                          {claim.status === "supported" ? <CheckCircle2 className="h-3 w-3" /> :
                            claim.status === "partially_supported" ? <AlertTriangle className="h-3 w-3" /> :
                              claim.status === "unsupported" ? <XCircle className="h-3 w-3" /> :
                                <HelpCircle className="h-3 w-3" />}
                        </span>

                        {/* Claim text */}
                        <p className="flex-1 min-w-0 text-foreground truncate">
                          {claim.claimText}
                        </p>

                        {/* Status label */}
                        <span className={cn(
                          "text-[10px] font-medium shrink-0",
                          CLAIM_TEXT_COLORS[claim.status]
                        )}>
                          {CLAIM_LABELS[claim.status]}
                        </span>

                        {/* Confidence */}
                        <span className={cn(
                          "text-[10px] font-mono font-medium shrink-0",
                          claim.confidence >= 0.8 ? "text-emerald-700" :
                            claim.confidence >= 0.5 ? "text-amber-700" : "text-red-700"
                        )}>
                          {Math.round(claim.confidence * 100)}%
                        </span>
                      </div>

                      {/* Review badge + Issues */}
                      <div className="flex gap-1 mt-1 ml-7 flex-wrap">
                        {/* Human review badge (Fix 4) */}
                        {claim.requiresReview && (
                          <span className={cn(
                            "inline-flex items-center gap-0.5 rounded px-1.5 py-0.5 text-[9px] font-semibold border",
                            claim.confidence < 0.30
                              ? "bg-red-100 text-red-800 border-red-300"
                              : "bg-amber-100 text-amber-800 border-amber-300"
                          )}>
                            <AlertTriangle className="h-2.5 w-2.5" />
                            {claim.confidence < 0.30 ? "Requires Review" : "Review Recommended"}
                          </span>
                        )}
                        {/* Issue badges */}
                        {claim.issues.filter(i => i !== "no_citation").map((issue, i) => (
                          <span
                            key={i}
                            className="inline-flex items-center rounded bg-red-50 px-1.5 py-0.5 text-[9px] font-medium text-red-700 border border-red-200/50"
                          >
                            {ISSUE_LABELS[issue] ?? issue}
                          </span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </>
      )}
    </div>
  )
}
