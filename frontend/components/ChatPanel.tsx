"use client"

import React, { useState, useRef, useEffect } from "react"
import { Send, Loader2, Copy, Check, ChevronDown, ChevronUp } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"
import type { QueryMessage, Citation } from "@/lib/types"

interface ChatPanelProps {
  messages: QueryMessage[]
  onSend: (message: string) => void
  isLoading?: boolean
  onSelectCitation?: (citations: Citation[]) => void
}

function ConfidenceBadge({ score }: { score: number }) {
  const color = score >= 80 ? "text-emerald-700 bg-emerald-50" : score >= 60 ? "text-amber-700 bg-amber-50" : "text-red-700 bg-red-50"
  return (
    <span className={cn("inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium", color)}>
      Confidence: {score}%
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
    <div className={cn("animate-slide-up", isUser ? "flex justify-end" : "")}>
      <div
        className={cn(
          "max-w-[85%] rounded-xl px-4 py-3",
          isUser
            ? "bg-accent text-white"
            : "bg-surface border border-border"
        )}
      >
        <div className="flex items-center gap-2 mb-1">
          <span className={cn("text-xs font-semibold", isUser ? "text-white/80" : "text-accent")}>
            {isUser ? "You" : "Veritas AI"}
          </span>
          <span className={cn("text-xs", isUser ? "text-white/50" : "text-muted")}>
            {new Date(message.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
          </span>
        </div>

        <p className={cn("text-sm leading-relaxed whitespace-pre-wrap", isUser ? "text-white" : "text-foreground")}>
          {message.content}
        </p>

        {!isUser && (
          <div className="mt-3 flex items-center gap-2 border-t border-border pt-3">
            {message.citations && message.citations.length > 0 && (
              <Button
                variant="ghost"
                size="sm"
                className="h-7 text-xs text-accent hover:text-accent-hover px-2"
                onClick={() => {
                  setShowSources(!showSources)
                  if (onViewCitations && message.citations) {
                    onViewCitations(message.citations)
                  }
                }}
              >
                {showSources ? <ChevronUp className="h-3 w-3 mr-1" /> : <ChevronDown className="h-3 w-3 mr-1" />}
                {message.citations.length} Sources
              </Button>
            )}
            {message.confidenceScore !== undefined && (
              <ConfidenceBadge score={message.confidenceScore} />
            )}
            <Button
              variant="ghost"
              size="sm"
              className="h-7 text-xs text-muted ml-auto px-2"
              onClick={handleCopy}
            >
              {copied ? <Check className="h-3 w-3 mr-1" /> : <Copy className="h-3 w-3 mr-1" />}
              {copied ? "Copied" : "Copy"}
            </Button>
          </div>
        )}

        {showSources && message.citations && (
          <div className="mt-2 space-y-1.5 animate-slide-up">
            {message.citations.map((c, i) => (
              <div key={i} className="flex items-center gap-2 rounded-md bg-white border border-border px-3 py-2 text-xs">
                <span className="font-medium text-foreground">{c.documentName}</span>
                <span className="text-muted">Page {c.pageNumber}</span>
                {c.section && <span className="text-muted">- {c.section}</span>}
                <span className="ml-auto text-accent font-medium">{Math.round(c.relevanceScore * 100)}%</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

export default function ChatPanel({ messages, onSend, isLoading, onSelectCitation }: ChatPanelProps) {
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
      {/* Messages */}
      <div className="flex-1 overflow-y-auto space-y-4 p-4">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center py-12">
            <div className="h-12 w-12 rounded-xl bg-accent/10 flex items-center justify-center mb-4">
              <Send className="h-6 w-6 text-accent" />
            </div>
            <h3 className="text-lg font-semibold text-foreground">Ask Veritas AI</h3>
            <p className="text-sm text-muted mt-1 max-w-sm">
              Ask questions about your uploaded documents. Every answer includes source citations and confidence scores.
            </p>
          </div>
        )}
        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} onViewCitations={onSelectCitation} />
        ))}
        {isLoading && (
          <div className="flex items-center gap-2 text-muted animate-fade-in">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span className="text-sm">Analyzing documents...</span>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} className="border-t border-border p-4">
        <div className="relative">
          <Textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about your documents..."
            className="pr-12 min-h-[48px] max-h-32 resize-none"
            rows={1}
          />
          <Button
            type="submit"
            size="icon"
            disabled={!input.trim() || isLoading}
            className="absolute right-2 bottom-2 h-8 w-8"
          >
            <Send className="h-4 w-4" />
          </Button>
        </div>
        <p className="text-xs text-muted mt-2">
          Press Enter to send, Shift+Enter for new line
        </p>
      </form>
    </div>
  )
}
