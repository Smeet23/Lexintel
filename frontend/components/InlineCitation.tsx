"use client"

import React, { useState } from "react"
import { ExternalLink, CheckCircle2, AlertTriangle, XCircle, HelpCircle, ShieldCheck, ShieldAlert, ShieldX } from "lucide-react"
import { cn, isSafeUrl } from "@/lib/utils"
import type { CitationVerification, VerificationStatus, CaseStatus } from "@/lib/types"

// ─── Status config ──────────────────────────────────────────

const STATUS_CONFIG: Record<VerificationStatus, {
  bg: string
  text: string
  border: string
  label: string
  Icon: React.ElementType
}> = {
  verified: {
    bg: "bg-emerald-100",
    text: "text-emerald-700",
    border: "border-emerald-300",
    label: "Verified",
    Icon: CheckCircle2,
  },
  partial: {
    bg: "bg-amber-100",
    text: "text-amber-700",
    border: "border-amber-300",
    label: "Partial Match",
    Icon: AlertTriangle,
  },
  unverified: {
    bg: "bg-red-100",
    text: "text-red-700",
    border: "border-red-300",
    label: "Unverified",
    Icon: XCircle,
  },
  not_found: {
    bg: "bg-red-200",
    text: "text-red-800",
    border: "border-red-400",
    label: "Not Found",
    Icon: XCircle,
  },
  unverifiable: {
    bg: "bg-gray-100",
    text: "text-gray-600",
    border: "border-gray-300",
    label: "Unable to Verify",
    Icon: HelpCircle,
  },
}

const CASE_STATUS_CONFIG: Record<CaseStatus, {
  bg: string
  text: string
  label: string
  Icon: React.ElementType
}> = {
  good_law: { bg: "bg-emerald-50", text: "text-emerald-700", label: "Good Law", Icon: ShieldCheck },
  caution: { bg: "bg-amber-50", text: "text-amber-700", label: "Caution", Icon: ShieldAlert },
  bad_law: { bg: "bg-red-50", text: "text-red-700", label: "Overruled", Icon: ShieldX },
  unknown: { bg: "bg-gray-50", text: "text-gray-500", label: "Status Unknown", Icon: HelpCircle },
}

// ─── Component ──────────────────────────────────────────────

interface InlineCitationProps {
  index: number
  verification?: CitationVerification
  onClick?: () => void
}

export default function InlineCitation({ index, verification, onClick }: InlineCitationProps) {
  const [showTooltip, setShowTooltip] = useState(false)

  const status = verification?.status ?? "unverifiable"
  const config = STATUS_CONFIG[status]

  const handleClick = () => {
    if (verification?.sourceUrl && isSafeUrl(verification.sourceUrl)) {
      window.open(verification.sourceUrl, "_blank", "noopener,noreferrer")
    }
    onClick?.()
  }

  return (
    <span className="relative inline-block">
      {/* Citation badge */}
      <button
        type="button"
        className={cn(
          "inline-flex items-center justify-center rounded-md border px-1 py-px",
          "text-[10px] font-semibold leading-none cursor-pointer transition-all duration-150",
          "hover:shadow-sm hover:scale-105 active:scale-95",
          "align-super -mx-px",
          config.bg, config.text, config.border,
          status === "not_found" && "line-through",
        )}
        onMouseEnter={() => setShowTooltip(true)}
        onMouseLeave={() => setShowTooltip(false)}
        onClick={handleClick}
        aria-label={`Citation ${index}: ${verification?.caseName ?? verification?.rawText ?? ""} — ${config.label}`}
      >
        {index}
      </button>

      {/* Hover tooltip */}
      {showTooltip && verification && (
        <div
          className={cn(
            "absolute z-50 w-72 rounded-xl border border-border bg-white shadow-lg",
            "p-3 text-left",
            "bottom-full left-1/2 -translate-x-1/2 mb-2",
            "animate-in fade-in-0 zoom-in-95 duration-150",
          )}
          onMouseEnter={() => setShowTooltip(true)}
          onMouseLeave={() => setShowTooltip(false)}
        >
          {/* Arrow */}
          <div className="absolute top-full left-1/2 -translate-x-1/2 -mt-px">
            <div className="w-2.5 h-2.5 bg-white border-r border-b border-border rotate-45 -translate-y-1.5" />
          </div>

          {/* Case name */}
          {verification.caseName && (
            <p className="text-[13px] font-semibold text-foreground leading-snug mb-1.5">
              {verification.caseName}
            </p>
          )}

          {/* Court + date */}
          {(verification.court || verification.date) && (
            <p className="text-[11px] text-muted-foreground mb-1.5">
              {[verification.court, verification.date].filter(Boolean).join(" · ")}
            </p>
          )}

          {/* Citation string */}
          <p className="text-[11px] font-mono text-muted-foreground bg-surface rounded-md px-2 py-1 mb-2">
            {verification.rawText}
          </p>

          {/* Status badge row */}
          <div className="flex items-center gap-2 mb-2">
            {/* Verification status */}
            <span className={cn(
              "inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[10px] font-medium",
              config.bg, config.text, config.border,
            )}>
              <config.Icon className="h-3 w-3" />
              {config.label}
            </span>

            {/* Case status */}
            {verification.caseStatus && verification.caseStatus !== "unknown" && (
              <span className={cn(
                "inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[10px] font-medium",
                CASE_STATUS_CONFIG[verification.caseStatus].bg,
                CASE_STATUS_CONFIG[verification.caseStatus].text,
                "border-current/20",
              )}>
                {React.createElement(CASE_STATUS_CONFIG[verification.caseStatus].Icon, { className: "h-3 w-3" })}
                {CASE_STATUS_CONFIG[verification.caseStatus].label}
              </span>
            )}
          </div>

          {/* Jurisdiction */}
          {verification.jurisdiction && (
            <div className="flex items-center justify-between text-[10px] text-muted-foreground mb-1">
              <span>Jurisdiction</span>
              <span className="font-medium">{verification.jurisdiction}</span>
            </div>
          )}

          {/* Citation count */}
          {verification.citationCount !== undefined && verification.citationCount > 0 && (
            <div className="flex items-center justify-between text-[10px] text-muted-foreground mb-1">
              <span>Cited by</span>
              <span className="font-medium">{verification.citationCount.toLocaleString()} cases</span>
            </div>
          )}

          {/* Confidence */}
          <div className="flex items-center justify-between text-[10px] text-muted-foreground mb-2">
            <span>Confidence</span>
            <span className={cn(
              "font-mono font-medium",
              verification.confidence >= 0.8 ? "text-emerald-700" :
                verification.confidence >= 0.5 ? "text-amber-700" : "text-red-700"
            )}>
              {Math.round(verification.confidence * 100)}%
            </span>
          </div>

          {/* CourtListener link */}
          {verification.sourceUrl && isSafeUrl(verification.sourceUrl) && (
            <a
              href={verification.sourceUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 text-[11px] text-blue-600 hover:text-blue-800 font-medium transition-colors"
              onClick={(e) => e.stopPropagation()}
            >
              <ExternalLink className="h-3 w-3" />
              View source
            </a>
          )}
        </div>
      )}
    </span>
  )
}
