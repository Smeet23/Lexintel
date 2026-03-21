"use client"

import React, { useState, useEffect, useMemo } from "react"
import { motion } from "framer-motion"
import { CheckCircle2, Loader2, AlertCircle, Square } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import type { ProgressEvent } from "@/lib/types"

const STAGES = [
  { key: "uploaded", label: "Upload" },
  { key: "downloading", label: "Download" },
  { key: "chunking", label: "Chunk" },
  { key: "embedding", label: "Embed" },
  { key: "indexing", label: "Index" },
  { key: "storing", label: "Store" },
  { key: "ready", label: "Ready" },
]

export default function MultiStageProgress({
  progress,
  onCancel,
  isCancelling,
}: {
  progress: ProgressEvent | null
  onCancel?: () => void
  isCancelling?: boolean
}) {
  // Track max progress to prevent backwards jumps
  const [maxProgress, setMaxProgress] = useState(0)

  useEffect(() => {
    if (!progress) {
      setMaxProgress(0)
      return
    }
    const incoming = progress.overall_progress ?? 0
    setMaxProgress((prev) => Math.max(prev, incoming))
  }, [progress])

  const currentStageIndex = useMemo(
    () => STAGES.findIndex((s) => s.key === progress?.stage),
    [progress?.stage]
  )

  if (!progress) return null

  const isError = !!progress.error
  const isReady = progress.stage === "ready"
  const displayPct = isReady ? 100 : maxProgress

  return (
    <div className="rounded-xl border border-amber-200/60 bg-amber-50/60 p-5 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-semibold text-amber-900">
            {isError ? "Processing Failed" : isReady ? "Processing Complete" : progress.message}
          </p>
          {progress.detail && !isReady && (
            <p className="text-xs text-amber-700 mt-0.5">
              {progress.detail}
              {progress.step > 0 && progress.total_steps > 0 && (
                <span className="ml-1.5 text-amber-600">
                  — Step {progress.step} of {progress.total_steps}
                </span>
              )}
            </p>
          )}
        </div>
        <div className="flex items-center gap-3">
          <span
            className={cn(
              "text-sm font-mono font-bold",
              isError ? "text-red-600" : isReady ? "text-emerald-600" : "text-amber-800"
            )}
          >
            {displayPct}%
          </span>
          {onCancel && !isReady && !isError && (
            <Button
              variant="outline"
              size="sm"
              className="shrink-0 border-destructive/50 text-destructive hover:bg-destructive/10 h-7 text-xs"
              onClick={onCancel}
              disabled={isCancelling}
            >
              {isCancelling ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <>
                  <Square className="h-3 w-3 mr-1" />
                  Cancel
                </>
              )}
            </Button>
          )}
        </div>
      </div>

      {/* Step indicators */}
      <div className="flex items-center gap-1">
        {STAGES.map((stage, idx) => {
          const isCompleted = idx < currentStageIndex
          const isActive = idx === currentStageIndex
          const isFailed = isError && isActive

          return (
            <React.Fragment key={stage.key}>
              <div
                className={cn(
                  "flex items-center justify-center w-6 h-6 rounded-full shrink-0 transition-colors",
                  isCompleted
                    ? "bg-emerald-500 text-white"
                    : isActive
                      ? isFailed
                        ? "bg-red-500 text-white"
                        : "bg-amber-500 text-white"
                      : "bg-amber-200/60 text-amber-500/50"
                )}
                title={stage.label}
              >
                {isCompleted ? (
                  <CheckCircle2 className="w-3.5 h-3.5" />
                ) : isFailed ? (
                  <AlertCircle className="w-3.5 h-3.5" />
                ) : isActive ? (
                  <Loader2 className="w-3 h-3 animate-spin" />
                ) : (
                  <span className="text-[9px] font-bold">{idx + 1}</span>
                )}
              </div>
              {idx < STAGES.length - 1 && (
                <div
                  className={cn(
                    "flex-1 h-0.5 rounded-full transition-colors",
                    isCompleted ? "bg-emerald-400" : "bg-amber-200/60"
                  )}
                />
              )}
            </React.Fragment>
          )
        })}
      </div>

      {/* Progress bar — GPU accelerated */}
      {!isReady && !isError && (
        <div className="h-2 w-full rounded-full bg-amber-200/60 overflow-hidden">
          <motion.div
            className="h-full rounded-full bg-gradient-to-r from-amber-400 to-amber-500"
            animate={{ scaleX: displayPct / 100 }}
            transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
            style={{ transformOrigin: "left", willChange: "transform" }}
            initial={{ scaleX: 0 }}
          />
        </div>
      )}

      {/* Success state */}
      {isReady && (
        <motion.div
          initial={{ scale: 0.95, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          className="flex items-center gap-2 text-emerald-600 text-sm font-medium"
        >
          <CheckCircle2 className="w-4 h-4" />
          Documents ready for analysis
        </motion.div>
      )}

      {/* Error state */}
      {isError && (
        <div className="flex items-center gap-2 text-red-600 text-sm">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>{progress.error}</span>
        </div>
      )}

      {/* Finalizing hint */}
      {displayPct > 90 && !isReady && !isError && (
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-[11px] text-amber-600"
        >
          Finalizing... this may take a moment
        </motion.p>
      )}
    </div>
  )
}
