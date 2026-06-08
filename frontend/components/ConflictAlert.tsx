"use client"

import React, { useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import {
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  Scale,
  FileText,
  Shield,
  Clock,
  Award,
} from "lucide-react"
import { cn } from "@/lib/utils"
import type { ConflictAnalysis, ConflictItem, ConflictCredibility } from "@/lib/types"

// --- Props ---------------------------------------------------------------

interface ConflictAlertProps {
  analysis: ConflictAnalysis
  className?: string
}

// --- Severity colors -----------------------------------------------------

const SEVERITY_STYLES = {
  high: {
    bg: "bg-red-50 border-red-200",
    badge: "bg-red-100 text-red-800 border-red-300",
    icon: "text-red-600",
    bar: "bg-red-500",
  },
  medium: {
    bg: "bg-amber-50 border-amber-200",
    badge: "bg-amber-100 text-amber-800 border-amber-300",
    icon: "text-amber-600",
    bar: "bg-amber-500",
  },
}

const AUTHORITY_LABELS: Record<string, string> = {
  constitution: "Constitution",
  statute: "Statute",
  regulation: "Regulation",
  case_law: "Case Law",
  supreme_court: "Supreme Court",
  high_court: "High Court",
  district_court: "District Court",
  tribunal: "Tribunal",
  commentary: "Commentary",
  practice_note: "Practice Note",
  opinion: "Opinion",
  unknown: "Unknown",
}

// --- Credibility breakdown -----------------------------------------------

function CredibilityBreakdown({ cred }: { cred: ConflictCredibility }) {
  return (
    <div className="mt-2 space-y-1 text-[11px]">
      <div className="flex items-center gap-1.5">
        <Shield className="h-3 w-3 text-slate-400" />
        <span className="text-slate-500">Authority:</span>
        <span className="font-medium text-slate-700">
          {AUTHORITY_LABELS[cred.authority_type] || cred.authority_type}
        </span>
        <span className="text-slate-400">({(cred.authority_weight * 100).toFixed(0)}%)</span>
      </div>
      <div className="flex items-center gap-1.5">
        <Clock className="h-3 w-3 text-slate-400" />
        <span className="text-slate-500">Recency:</span>
        <span className="font-medium text-slate-700">{(cred.recency_weight * 100).toFixed(0)}%</span>
      </div>
      <div className="flex items-center gap-1.5">
        <Award className="h-3 w-3 text-slate-400" />
        <span className="text-slate-500">Overall:</span>
        <span className="font-semibold text-slate-800">{(cred.score * 100).toFixed(0)}%</span>
      </div>
    </div>
  )
}

// --- Single conflict card ------------------------------------------------

function ConflictCard({ conflict }: { conflict: ConflictItem }) {
  const [expanded, setExpanded] = useState(false)
  const styles = SEVERITY_STYLES[conflict.severity]
  const recommended = conflict.recommended_source === "a" ? conflict.chunk_a : conflict.chunk_b
  const other = conflict.recommended_source === "a" ? conflict.chunk_b : conflict.chunk_a

  return (
    <div className={cn("rounded-lg border p-3", styles.bg)}>
      {/* Header */}
      <button
        type="button"
        aria-expanded={expanded}
        onClick={() => setExpanded(!expanded)}
        className="flex w-full items-start justify-between text-left"
      >
        <div className="flex items-start gap-2">
          <Scale className={cn("mt-0.5 h-4 w-4 flex-shrink-0", styles.icon)} />
          <div>
            <div className="flex items-center gap-2">
              <span className={cn("inline-flex items-center rounded border px-1.5 py-0.5 text-[10px] font-semibold uppercase", styles.badge)}>
                {conflict.severity}
              </span>
              <span className="text-xs text-slate-600">
                {(conflict.contradiction_score * 100).toFixed(0)}% contradiction
              </span>
            </div>
            <p className="mt-1 text-xs text-slate-700 leading-relaxed">
              {conflict.explanation}
            </p>
          </div>
        </div>
        {expanded ? (
          <ChevronUp className="h-4 w-4 text-slate-400 flex-shrink-0" />
        ) : (
          <ChevronDown className="h-4 w-4 text-slate-400 flex-shrink-0" />
        )}
      </button>

      {/* Side-by-side comparison */}
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="mt-3 grid grid-cols-2 gap-3">
              {/* Recommended source */}
              <div className="rounded-md border border-emerald-200 bg-emerald-50/50 p-2.5">
                <div className="flex items-center gap-1.5 mb-1.5">
                  <FileText className="h-3.5 w-3.5 text-emerald-600" />
                  <span className="text-[11px] font-semibold text-emerald-800 truncate">
                    {recommended.document_name}
                  </span>
                </div>
                <p className="text-[10px] text-slate-500 mb-1">
                  p. {recommended.page_num}
                  {recommended.section_name ? `, ${recommended.section_name}` : ""}
                </p>
                <p className="text-[11px] text-slate-700 leading-relaxed line-clamp-4">
                  {recommended.content_snippet}
                </p>
                <CredibilityBreakdown cred={recommended.credibility} />
                <div className="mt-1.5">
                  <span className="inline-flex items-center rounded bg-emerald-100 px-1.5 py-0.5 text-[10px] font-medium text-emerald-700 border border-emerald-200">
                    Recommended
                  </span>
                </div>
              </div>

              {/* Other source */}
              <div className="rounded-md border border-slate-200 bg-white/50 p-2.5">
                <div className="flex items-center gap-1.5 mb-1.5">
                  <FileText className="h-3.5 w-3.5 text-slate-500" />
                  <span className="text-[11px] font-semibold text-slate-700 truncate">
                    {other.document_name}
                  </span>
                </div>
                <p className="text-[10px] text-slate-500 mb-1">
                  p. {other.page_num}
                  {other.section_name ? `, ${other.section_name}` : ""}
                </p>
                <p className="text-[11px] text-slate-700 leading-relaxed line-clamp-4">
                  {other.content_snippet}
                </p>
                <CredibilityBreakdown cred={other.credibility} />
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

// --- Main component ------------------------------------------------------

export default function ConflictAlert({ analysis, className }: ConflictAlertProps) {
  const [collapsed, setCollapsed] = useState(false)
  const { summary, conflicts } = analysis

  if (!conflicts.length) return null

  return (
    <div className={cn("rounded-xl border border-amber-200 bg-amber-50/30 p-3", className)}>
      {/* Summary header */}
      <button
        type="button"
        aria-expanded={!collapsed}
        onClick={() => setCollapsed(!collapsed)}
        className="flex w-full items-center justify-between text-left"
      >
        <div className="flex items-center gap-2">
          <AlertTriangle className="h-4 w-4 text-amber-600" />
          <span className="text-sm font-semibold text-amber-900">
            {summary.total_conflicts} Source Conflict{summary.total_conflicts !== 1 ? "s" : ""} Detected
          </span>
          {summary.high_severity > 0 && (
            <span className="inline-flex items-center rounded border border-red-300 bg-red-100 px-1.5 py-0.5 text-[10px] font-semibold text-red-800">
              {summary.high_severity} HIGH
            </span>
          )}
        </div>
        {collapsed ? (
          <ChevronDown className="h-4 w-4 text-amber-500" />
        ) : (
          <ChevronUp className="h-4 w-4 text-amber-500" />
        )}
      </button>

      {/* Action text */}
      <p className="mt-1 text-xs text-amber-700 pl-6">
        {summary.recommended_action}
      </p>

      {/* Conflict cards */}
      <AnimatePresence>
        {!collapsed && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="mt-3 space-y-2">
              {conflicts.map((conflict) => (
                <ConflictCard key={conflict.id} conflict={conflict} />
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
