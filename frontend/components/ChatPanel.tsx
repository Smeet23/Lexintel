"use client"

import React, { useState, useRef, useEffect, useMemo } from "react"
import { motion, AnimatePresence } from "framer-motion"
import {
  Send,
  Loader2,
  Copy,
  Check,
  ChevronDown,
  ChevronUp,
  Scale,
  Trash2,
  MessageSquarePlus,
  AlertTriangle,
  Lightbulb,
  AlertCircle,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { cn } from "@/lib/utils"
import type { QueryMessage, Citation, CitationVerification, ConflictAnalysis } from "@/lib/types"
import InlineCitation from "@/components/InlineCitation"
import VerificationBar from "@/components/VerificationBar"
import ConflictAlert from "@/components/ConflictAlert"

// ─── Helpers ────────────────────────────────────────────────

function formatMessageDate(timestamp: string): string {
  const date = new Date(timestamp)
  const now = new Date()
  const diff = now.getTime() - date.getTime()
  const days = Math.floor(diff / 86400000)

  if (days === 0) return "Today"
  if (days === 1) return "Yesterday"
  if (days < 7) return date.toLocaleDateString([], { weekday: "long" })
  return date.toLocaleDateString([], { month: "short", day: "numeric", year: "numeric" })
}

function ConfidenceBadge({ score }: { score: number }) {
  const color = score >= 80
    ? "text-emerald-700 bg-emerald-50 border-emerald-200/60"
    : score >= 60
      ? "text-amber-700 bg-amber-50 border-amber-200/60"
      : "text-red-700 bg-red-50 border-red-200/60"
  return (
    <span className={cn("inline-flex items-center rounded-md border px-2 py-0.5 text-[11px] font-medium", color)}>
      {score}% confidence
    </span>
  )
}

// ─── Date Separator ─────────────────────────────────────────

function DateSeparator({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-3 py-2">
      <div className="flex-1 h-px bg-border" />
      <span className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground/60 select-none">
        {label}
      </span>
      <div className="flex-1 h-px bg-border" />
    </div>
  )
}

// ─── Delete Confirmation Inline ──────────────────────────────

function DeleteConfirmInline({
  onConfirm,
  onCancel,
  isDeleting,
}: {
  onConfirm: () => void
  onCancel: () => void
  isDeleting?: boolean
}) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.95 }}
      transition={{ duration: 0.15 }}
      className="flex items-center gap-2 mt-2 p-2 rounded-lg bg-red-50 border border-red-200/60"
    >
      <AlertTriangle className="h-3.5 w-3.5 text-red-500 shrink-0" />
      <span className="text-[11px] text-red-700 flex-1">Delete this Q&A?</span>
      <Button
        variant="ghost"
        size="sm"
        className="h-5 text-[10px] px-1.5 text-muted-foreground hover:text-foreground"
        onClick={onCancel}
      >
        Cancel
      </Button>
      <Button
        size="sm"
        className="h-5 text-[10px] px-2 bg-red-600 hover:bg-red-700 text-white"
        onClick={onConfirm}
        disabled={isDeleting}
      >
        {isDeleting ? <Loader2 className="h-3 w-3 animate-spin" /> : "Delete"}
      </Button>
    </motion.div>
  )
}

// ─── Inline Citation Parser ──────────────────────────────────

function parseMessageWithCitations(
  content: string,
  verifiedCitations?: CitationVerification[],
): React.ReactNode[] {
  // Match [N] patterns (numbered citations)
  const parts: React.ReactNode[] = []
  const regex = /\[(\d+)\]/g
  let lastIndex = 0
  let match: RegExpExecArray | null

  while ((match = regex.exec(content)) !== null) {
    // Add text before the citation
    if (match.index > lastIndex) {
      parts.push(content.slice(lastIndex, match.index))
    }

    const citationIndex = parseInt(match[1], 10)
    const verification = verifiedCitations?.find(c => c.index === citationIndex)

    parts.push(
      <InlineCitation
        key={`cite-${match.index}-${citationIndex}`}
        index={citationIndex}
        verification={verification}
      />
    )

    lastIndex = match.index + match[0].length
  }

  // Add remaining text
  if (lastIndex < content.length) {
    parts.push(content.slice(lastIndex))
  }

  return parts.length > 0 ? parts : [content]
}

// ─── Message Bubble ──────────────────────────────────────────

function MessageBubble({
  message,
  onViewCitations,
  onCitationClick,
  onDelete,
  isDeletingMessage,
}: {
  message: QueryMessage
  onViewCitations?: (citations: Citation[]) => void
  onCitationClick?: (citation: Citation) => void
  onDelete?: () => void
  isDeletingMessage?: boolean
}) {
  const [copied, setCopied] = useState(false)
  const [showSources, setShowSources] = useState(false)
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)
  const isUser = message.role === "user"

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <motion.div
      id={message.queryId ? `msg-${message.queryId}` : undefined}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
      className={cn("group transition-shadow duration-500", isUser ? "flex justify-end" : "")}
    >
      <div className={cn(
        "max-w-[85%] rounded-xl px-4 py-3 relative",
        isUser
          ? "bg-primary text-primary-foreground"
          : "bg-white border border-border shadow-sm"
      )}>
        {/* Delete button — top-right corner, hover only */}
        {!isUser && onDelete && message.queryId && !showDeleteConfirm && (
          <button
            type="button"
            className="absolute -top-2 -right-2 h-6 w-6 rounded-full bg-white border border-border shadow-sm flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity hover:bg-red-50 hover:border-red-200 hover:text-red-600 text-muted-foreground/50"
            onClick={() => setShowDeleteConfirm(true)}
            title="Delete this Q&A"
          >
            <Trash2 className="h-3 w-3" />
          </button>
        )}
        <div className="flex items-center gap-1.5 mb-2">
          {!isUser && (
            <div className="flex h-5 w-5 items-center justify-center rounded-md bg-surface shrink-0">
              <Scale className="h-3 w-3 text-foreground" />
            </div>
          )}
          <span className={cn("text-[10px] font-semibold uppercase tracking-[0.08em]", isUser ? "text-white/60" : "text-muted")}>
            {isUser ? "You" : "LexIntel"}
          </span>
          <span className={cn("text-[10px]", isUser ? "text-white/35" : "text-muted-foreground")}>
            {new Date(message.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
          </span>
        </div>

        <div className={cn("text-[14px] leading-relaxed whitespace-pre-wrap", isUser ? "text-primary-foreground" : "text-foreground")}>
          {!isUser && message.verifiedCitations && message.verifiedCitations.length > 0
            ? parseMessageWithCitations(message.content, message.verifiedCitations)
            : message.content
          }
        </div>

        {/* Issue Analysis Badges */}
        {!isUser && message.issueAnalysis && message.issueAnalysis.issues.length > 0 && (
          <div className="flex flex-wrap items-center gap-1.5 mt-2">
            <Lightbulb className="h-3.5 w-3.5 text-amber-500 shrink-0" />
            <span className="text-[11px] font-medium text-muted-foreground mr-1">Issues:</span>
            {message.issueAnalysis.issues.map((issue, i) => (
              <span
                key={i}
                className={cn(
                  "inline-flex items-center rounded-md px-2 py-0.5 text-[10px] font-medium border",
                  issue.confidence >= 0.7
                    ? "bg-blue-50 text-blue-700 border-blue-200/60"
                    : issue.confidence >= 0.4
                      ? "bg-slate-50 text-slate-600 border-slate-200/60"
                      : "bg-gray-50 text-gray-500 border-gray-200/60"
                )}
                title={`${issue.issue}\nConfidence: ${(issue.confidence * 100).toFixed(0)}%\nKey facts: ${issue.keyFacts.join(", ")}`}
              >
                {issue.domain}
                <span className="ml-1 opacity-60">{(issue.confidence * 100).toFixed(0)}%</span>
              </span>
            ))}
            {message.issueAnalysis.missingInformation.length > 0 && (
              <span
                className="inline-flex items-center gap-0.5 rounded-md px-2 py-0.5 text-[10px] font-medium bg-amber-50 text-amber-700 border border-amber-200/60"
                title={`Missing: ${message.issueAnalysis.missingInformation.join("; ")}`}
              >
                <AlertCircle className="h-2.5 w-2.5" />
                {message.issueAnalysis.missingInformation.length} missing info
              </span>
            )}
          </div>
        )}

        {/* Verification bar (citations + claims) */}
        {!isUser && ((message.verificationSummary && message.verifiedCitations && message.verificationSummary.total > 0) || (message.claimVerificationSummary && message.claimVerificationSummary.total > 0)) && (
          <div className="mt-2">
            <VerificationBar
              summary={message.verificationSummary ?? { total: 0, verified: 0, partial: 0, unverified: 0, not_found: 0, unverifiable: 0 }}
              citations={message.verifiedCitations ?? []}
              claimSummary={message.claimVerificationSummary}
              claims={message.verifiedClaims}
            />
          </div>
        )}

        {/* Conflict alert */}
        {!isUser && message.conflictAnalysis?.has_conflicts && (
          <ConflictAlert
            analysis={message.conflictAnalysis}
            className="mt-2"
          />
        )}

        {!isUser && (
          <div className="mt-3 flex items-center gap-2 border-t border-border pt-3">
            {message.citations && message.citations.length > 0 && (
              <Button
                variant="ghost"
                size="sm"
                className="h-6 text-[11px] text-muted hover:text-foreground px-2"
                onClick={() => {
                  setShowSources(!showSources)
                  if (onViewCitations && message.citations) {
                    onViewCitations(message.citations)
                  }
                }}
              >
                {showSources ? <ChevronUp className="h-3 w-3 mr-1" /> : <ChevronDown className="h-3 w-3 mr-1" />}
                {message.citations.length} sources
              </Button>
            )}
            {message.confidenceScore !== undefined && (
              <ConfidenceBadge score={message.confidenceScore} />
            )}
            <Button
              variant="ghost"
              size="sm"
              className="h-6 text-[11px] text-muted ml-auto px-2"
              onClick={handleCopy}
            >
              {copied ? <Check className="h-3 w-3 mr-1" /> : <Copy className="h-3 w-3 mr-1" />}
              {copied ? "Copied" : "Copy"}
            </Button>
          </div>
        )}

        {/* Inline delete confirmation */}
        <AnimatePresence>
          {showDeleteConfirm && !isUser && (
            <DeleteConfirmInline
              onConfirm={() => {
                onDelete?.()
                setShowDeleteConfirm(false)
              }}
              onCancel={() => setShowDeleteConfirm(false)}
              isDeleting={isDeletingMessage}
            />
          )}
        </AnimatePresence>

        <AnimatePresence>
          {showSources && message.citations && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.25 }}
              className="mt-2 space-y-1 overflow-hidden"
            >
              {message.citations.map((c, i) => (
                <button
                  key={i}
                  type="button"
                  onClick={() => onCitationClick?.(c)}
                  className="w-full flex items-center gap-2 rounded-lg bg-surface px-3 py-2 text-[12px] text-left hover:bg-surface-hover transition-colors cursor-pointer border border-transparent hover:border-border"
                >
                  <span className="font-medium text-foreground">{c.documentName}</span>
                  <span className="text-muted">p. {c.pageNumber}</span>
                  {c.section && <span className="text-muted">&middot; {c.section}</span>}
                  <span className="ml-auto text-foreground font-mono text-[11px]">{Math.round(c.relevanceScore * 100)}%</span>
                </button>
              ))}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  )
}

// ─── Main Chat Panel ─────────────────────────────────────────

export default function ChatPanel({
  messages,
  onSend,
  isLoading,
  onSelectCitation,
  onCitationClick,
  onNewChat,
  onClearHistory,
  onDeleteMessage,
  isClearingHistory,
  isDeletingMessage,
  hasHistory,
}: {
  messages: QueryMessage[]
  onSend: (message: string, includeLegalResearch: boolean) => void
  isLoading?: boolean
  onSelectCitation?: (citations: Citation[]) => void
  onCitationClick?: (citation: Citation) => void
  onNewChat?: () => void
  onClearHistory?: () => void
  onDeleteMessage?: (queryId: string) => void
  isClearingHistory?: boolean
  isDeletingMessage?: boolean
  hasHistory?: boolean
}) {
  const [input, setInput] = useState("")
  const [includeLegalResearch, setIncludeLegalResearch] = useState(false)
  const [showClearConfirm, setShowClearConfirm] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const questionCount = useMemo(
    () => messages.filter(m => m.role === "user").length,
    [messages]
  )

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  // Compute date separators
  type DateItem = { type: "date"; label: string }
  type MsgItem = { type: "message"; message: QueryMessage }
  const messagesWithDates = useMemo(() => {
    const result: (DateItem | MsgItem)[] = []
    let lastDate = ""
    for (const msg of messages) {
      if (msg.role === "user") {
        const dateLabel = formatMessageDate(msg.timestamp)
        if (dateLabel !== lastDate) {
          lastDate = dateLabel
          result.push({ type: "date", label: dateLabel })
        }
      }
      result.push({ type: "message", message: msg })
    }
    return result
  }, [messages])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (input.trim() && !isLoading) {
      onSend(input.trim(), includeLegalResearch)
      setInput("")
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  const handleClearConfirm = () => {
    onClearHistory?.()
    setShowClearConfirm(false)
  }

  // Is this a "new chat" state vs "truly empty"?
  const isNewChatState = messages.length === 0 && hasHistory

  return (
    <div className="flex flex-col h-full">
      {/* ── Chat Header ─────────────────────────── */}
      <div className="flex items-center justify-between border-b border-border px-5 py-3 shrink-0 bg-white/80 backdrop-blur-sm">
        <div className="flex items-center gap-2.5">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-primary/10">
            <Scale className="h-3.5 w-3.5 text-primary" />
          </div>
          <div>
            <span className="text-[13px] font-semibold text-foreground leading-none">
              {isNewChatState ? "New Conversation" : "Chat"}
            </span>
            {questionCount > 0 && (
              <span className="ml-2 text-[10px] text-muted-foreground bg-surface/80 px-1.5 py-0.5 rounded-full font-medium">
                {questionCount} {questionCount === 1 ? "question" : "questions"}
              </span>
            )}
          </div>
        </div>

        <div className="flex items-center gap-1">
          {onNewChat && (
            <Button
              variant="ghost"
              size="sm"
              className="h-7 text-[11px] text-muted-foreground hover:text-foreground px-2.5 gap-1.5 rounded-lg"
              onClick={onNewChat}
              disabled={messages.length === 0 && !hasHistory}
              title="Start a new conversation"
            >
              <MessageSquarePlus className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">New Chat</span>
            </Button>
          )}
          {onClearHistory && (questionCount > 0 || hasHistory) && (
            <Button
              variant="ghost"
              size="sm"
              className={cn(
                "h-7 text-[11px] px-2.5 gap-1.5 rounded-lg transition-colors",
                showClearConfirm
                  ? "text-red-700 bg-red-50"
                  : "text-muted-foreground hover:text-red-600 hover:bg-red-50"
              )}
              onClick={() => setShowClearConfirm(true)}
              disabled={isClearingHistory}
              title="Clear all chat history"
            >
              {isClearingHistory ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <Trash2 className="h-3.5 w-3.5" />
              )}
              <span className="hidden sm:inline">Clear All</span>
            </Button>
          )}
        </div>
      </div>

      {/* ── Clear All Confirmation Banner ─────────── */}
      <AnimatePresence>
        {showClearConfirm && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden shrink-0"
          >
            <div className="flex items-center gap-3 px-5 py-3 bg-red-50 border-b border-red-200/60">
              <AlertTriangle className="h-4 w-4 text-red-500 shrink-0" />
              <div className="flex-1 min-w-0">
                <p className="text-[12px] font-medium text-red-800">
                  Delete all chat history?
                </p>
                <p className="text-[11px] text-red-600 mt-0.5">
                  This will permanently remove all {questionCount} question{questionCount !== 1 ? "s" : ""} and answers. This cannot be undone.
                </p>
              </div>
              <div className="flex items-center gap-2 shrink-0">
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-7 text-[11px] px-3"
                  onClick={() => setShowClearConfirm(false)}
                >
                  Cancel
                </Button>
                <Button
                  size="sm"
                  className="h-7 text-[11px] px-3 bg-red-600 hover:bg-red-700 text-white shadow-sm"
                  onClick={handleClearConfirm}
                  disabled={isClearingHistory}
                >
                  {isClearingHistory ? (
                    <>
                      <Loader2 className="h-3 w-3 animate-spin mr-1" />
                      Deleting...
                    </>
                  ) : (
                    "Delete All"
                  )}
                </Button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Messages Area ─────────────────────────── */}
      <div className="flex-1 overflow-y-auto space-y-4 p-6">
        {/* Empty state: first time / no history */}
        {messages.length === 0 && !isNewChatState && (
          <div className="flex flex-col items-center justify-center h-full text-center py-12">
            <div className="h-14 w-14 rounded-2xl bg-gradient-to-br from-primary/10 to-primary/5 flex items-center justify-center mb-5 shadow-sm border border-primary/10">
              <Scale className="h-6 w-6 text-primary/70" />
            </div>
            <h3 className="font-display text-[18px] text-foreground font-semibold">Ask LexIntel</h3>
            <p className="text-[13px] text-muted mt-2 max-w-sm leading-relaxed">
              Ask questions about your uploaded documents. Every answer includes source citations and confidence scores.
            </p>
          </div>
        )}

        {/* Empty state: new chat (history exists but cleared locally) */}
        {isNewChatState && (
          <div className="flex flex-col items-center justify-center h-full text-center py-12">
            <div className="h-14 w-14 rounded-2xl bg-gradient-to-br from-emerald-50 to-emerald-100/50 flex items-center justify-center mb-5 shadow-sm border border-emerald-200/40">
              <MessageSquarePlus className="h-6 w-6 text-emerald-600/70" />
            </div>
            <h3 className="font-display text-[18px] text-foreground font-semibold">New Conversation</h3>
            <p className="text-[13px] text-muted mt-2 max-w-sm leading-relaxed">
              Start a fresh conversation about your documents.
            </p>
          </div>
        )}

        {/* Messages with date separators */}
        {messagesWithDates.map((item, idx) => {
          if (item.type === "date") {
            return <DateSeparator key={`date-${idx}`} label={item.label} />
          }
          const msg = item.message
          return (
            <MessageBubble
              key={msg.id}
              message={msg}
              onViewCitations={onSelectCitation}
              onCitationClick={onCitationClick}
              onDelete={
                msg.queryId && onDeleteMessage
                  ? () => onDeleteMessage(msg.queryId!)
                  : undefined
              }
              isDeletingMessage={isDeletingMessage}
            />
          )
        })}
        {isLoading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex items-center gap-2.5 text-muted px-1 py-2"
          >
            <div className="flex h-6 w-6 items-center justify-center rounded-lg bg-surface">
              <Loader2 className="h-3.5 w-3.5 animate-spin text-primary" />
            </div>
            <span className="text-[13px]">Analyzing documents...</span>
          </motion.div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* ── Input Area ────────────────────────────── */}
      <form onSubmit={handleSubmit} className="border-t border-border p-4 bg-white/80 backdrop-blur-sm">
        <div className="relative">
          <Textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about your documents..."
            className="pr-12 min-h-[48px] max-h-32 resize-none text-[14px] rounded-xl"
            rows={1}
          />
          <Button
            type="submit"
            size="icon"
            disabled={!input.trim() || isLoading}
            className="absolute right-2 bottom-2 h-8 w-8 rounded-lg"
          >
            <Send className="h-3.5 w-3.5" />
          </Button>
        </div>
        <div className="flex items-center justify-between mt-2">
          <p className="text-[11px] text-muted-foreground">
            Enter to send &middot; Shift+Enter for new line
          </p>
          <label className="flex items-center gap-2 cursor-pointer select-none">
            <span className="text-[11px] text-muted-foreground">Include Legal Research</span>
            <button
              type="button"
              role="switch"
              aria-checked={includeLegalResearch}
              onClick={() => setIncludeLegalResearch(!includeLegalResearch)}
              className={cn(
                "relative inline-flex h-5 w-9 shrink-0 rounded-full border-2 border-transparent transition-colors",
                includeLegalResearch ? "bg-primary" : "bg-muted/30"
              )}
            >
              <span
                className={cn(
                  "pointer-events-none block h-4 w-4 rounded-full bg-white shadow-sm transition-transform",
                  includeLegalResearch ? "translate-x-4" : "translate-x-0"
                )}
              />
            </button>
          </label>
        </div>
      </form>
    </div>
  )
}
