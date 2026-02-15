"use client"

import React, { useState, useRef, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Send, Loader2, Copy, Check, ChevronDown, ChevronUp, Scale } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { cn } from "@/lib/utils"
import type { QueryMessage, Citation } from "@/lib/types"

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

function MessageBubble({
  message,
  onViewCitations,
}: {
  message: QueryMessage
  onViewCitations?: (citations: Citation[]) => void
}) {
  const [copied, setCopied] = useState(false)
  const [showSources, setShowSources] = useState(false)
  const isUser = message.role === "user"

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
      className={cn(isUser ? "flex justify-end" : "")}
    >
      <div className={cn(
        "max-w-[85%] rounded-xl px-4 py-3",
        isUser
          ? "bg-primary text-primary-foreground"
          : "bg-white border border-border shadow-sm"
      )}>
        <div className="flex items-center gap-2 mb-1.5">
          {!isUser && (
            <div className="flex h-5 w-5 items-center justify-center rounded-md bg-surface">
              <Scale className="h-3 w-3 text-foreground" />
            </div>
          )}
          <span className={cn("text-[11px] font-semibold uppercase tracking-[0.05em]", isUser ? "text-white/50" : "text-muted-foreground")}>
            {isUser ? "You" : "LexIntel"}
          </span>
          <span className={cn("text-[11px]", isUser ? "text-white/30" : "text-muted-foreground")}>
            {new Date(message.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
          </span>
        </div>

        <p className={cn("text-[14px] leading-relaxed whitespace-pre-wrap", isUser ? "text-primary-foreground" : "text-foreground")}>
          {message.content}
        </p>

        {!isUser && (
          <div className="mt-3 flex items-center gap-2 border-t border-border pt-3">
            {message.citations && message.citations.length > 0 && (
              <Button
                variant="ghost"
                size="sm"
                className="h-6 text-[11px] text-muted-foreground hover:text-foreground px-2"
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
              className="h-6 text-[11px] text-muted-foreground ml-auto px-2"
              onClick={handleCopy}
            >
              {copied ? <Check className="h-3 w-3 mr-1" /> : <Copy className="h-3 w-3 mr-1" />}
              {copied ? "Copied" : "Copy"}
            </Button>
          </div>
        )}

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
                <div key={i} className="flex items-center gap-2 rounded-lg bg-surface px-3 py-2 text-[12px]">
                  <span className="font-medium text-foreground">{c.documentName}</span>
                  <span className="text-muted-foreground">p. {c.pageNumber}</span>
                  {c.section && <span className="text-muted-foreground">&middot; {c.section}</span>}
                  <span className="ml-auto text-foreground font-mono text-[11px]">{Math.round(c.relevanceScore * 100)}%</span>
                </div>
              ))}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  )
}

export default function ChatPanel({
  messages,
  onSend,
  isLoading,
  onSelectCitation,
}: {
  messages: QueryMessage[]
  onSend: (message: string) => void
  isLoading?: boolean
  onSelectCitation?: (citations: Citation[]) => void
}) {
  const [input, setInput] = useState("")
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (input.trim() && !isLoading) {
      onSend(input.trim())
      setInput("")
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto space-y-4 p-6">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center py-12">
            <div className="h-12 w-12 rounded-xl bg-surface flex items-center justify-center mb-4">
              <Scale className="h-5 w-5 text-muted-foreground" />
            </div>
            <h3 className="font-display text-[18px] text-foreground">Ask LexIntel</h3>
            <p className="text-[13px] text-muted-foreground mt-1.5 max-w-sm leading-relaxed">
              Ask questions about your uploaded documents. Every answer includes source citations and confidence scores.
            </p>
          </div>
        )}
        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} onViewCitations={onSelectCitation} />
        ))}
        {isLoading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex items-center gap-2 text-muted-foreground px-1"
          >
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            <span className="text-[13px]">Analyzing documents...</span>
          </motion.div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSubmit} className="border-t border-border p-4">
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
        <p className="text-[11px] text-muted-foreground mt-2">
          Enter to send &middot; Shift+Enter for new line
        </p>
      </form>
    </div>
  )
}
