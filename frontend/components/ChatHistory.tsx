"use client"

import React, { useState, useMemo } from "react"
import { motion, AnimatePresence } from "framer-motion"
import {
  Search,
  X,
  MessageSquare,
  Trash2,
  Loader2,
  PanelLeftClose,
  PanelLeftOpen,
  Plus,
  MessageSquarePlus,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import type { ConversationItem } from "@/lib/types"

// ─── Helpers ────────────────────────────────────────────────

function formatRelativeShort(dateStr: string): string {
  const date = new Date(dateStr)
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffMin = Math.floor(diffMs / 60000)
  const diffHr = Math.floor(diffMs / 3600000)
  const diffDay = Math.floor(diffMs / 86400000)

  if (diffMin < 1) return "now"
  if (diffMin < 60) return `${diffMin}m`
  if (diffHr < 24) return `${diffHr}h`
  if (diffDay === 1) return "1d"
  if (diffDay < 7) return `${diffDay}d`
  return date.toLocaleDateString([], { month: "short", day: "numeric" })
}

function getDateGroup(dateStr: string): string {
  const date = new Date(dateStr)
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffDay = Math.floor(diffMs / 86400000)

  if (diffDay === 0) return "Today"
  if (diffDay === 1) return "Yesterday"
  if (diffDay < 7) return "This Week"
  if (diffDay < 30) return "This Month"
  return date.toLocaleDateString([], { month: "long", year: "numeric" })
}

function highlightMatch(text: string, query: string): React.ReactNode {
  if (!query.trim()) return text
  const idx = text.toLowerCase().indexOf(query.toLowerCase())
  if (idx === -1) return text
  return (
    <>
      {text.slice(0, idx)}
      <mark className="bg-yellow-200/80 text-foreground rounded-sm px-0.5">
        {text.slice(idx, idx + query.length)}
      </mark>
      {text.slice(idx + query.length)}
    </>
  )
}

// ─── Props ───────────────────────────────────────────────────

export interface ChatHistoryProps {
  conversations: ConversationItem[]
  activeConversationId: string | null
  onSelect: (id: string) => void
  onNewChat: () => void
  onDelete: (id: string) => void
  isDeleting?: boolean
  isCollapsed: boolean
  onToggleCollapse: () => void
  searchQuery: string
  onSearchChange: (q: string) => void
}

// ─── Component ───────────────────────────────────────────────

export default function ChatHistory({
  conversations,
  activeConversationId,
  onSelect,
  onNewChat,
  onDelete,
  isDeleting,
  isCollapsed,
  onToggleCollapse,
  searchQuery,
  onSearchChange,
}: ChatHistoryProps) {
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null)

  const filtered = useMemo(() => {
    if (!searchQuery.trim()) return conversations
    const q = searchQuery.toLowerCase()
    return conversations.filter(
      (conv) =>
        (conv.title ?? "").toLowerCase().includes(q) ||
        (conv.last_message_preview ?? "").toLowerCase().includes(q)
    )
  }, [conversations, searchQuery])

  const grouped = useMemo(() => {
    const groups: { label: string; items: ConversationItem[] }[] = []
    let currentLabel = ""
    for (const conv of filtered) {
      const label = getDateGroup(conv.updated_at)
      if (label !== currentLabel) {
        currentLabel = label
        groups.push({ label, items: [conv] })
      } else {
        groups[groups.length - 1].items.push(conv)
      }
    }
    return groups
  }, [filtered])

  // ── Collapsed state ──────────────────────────────────────────
  if (isCollapsed) {
    return (
      <div className="flex flex-col items-center py-3 px-1 border-r border-border bg-surface/30 w-10 shrink-0">
        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8 text-muted-foreground hover:text-foreground"
          onClick={onToggleCollapse}
          title="Show conversations"
        >
          <PanelLeftOpen className="h-4 w-4" />
        </Button>
        {conversations.length > 0 && (
          <span className="mt-2 text-[9px] text-muted-foreground font-medium bg-surface rounded-full px-1.5 py-0.5">
            {conversations.length}
          </span>
        )}
      </div>
    )
  }

  return (
    <div className="flex flex-col w-[240px] shrink-0 border-r border-border bg-surface/30 overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2.5 border-b border-border shrink-0">
        <div className="flex items-center gap-1.5">
          <MessageSquare className="h-3.5 w-3.5 text-muted-foreground" />
          <span className="text-[12px] font-semibold text-foreground">Chats</span>
          {conversations.length > 0 && (
            <span className="text-[10px] text-muted-foreground bg-white rounded-full px-1.5 py-0.5 border border-border font-medium">
              {conversations.length}
            </span>
          )}
        </div>
        <div className="flex items-center gap-0.5">
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6 text-muted-foreground hover:text-foreground"
            onClick={onNewChat}
            title="New chat"
          >
            <Plus className="h-3.5 w-3.5" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6 text-muted-foreground hover:text-foreground"
            onClick={onToggleCollapse}
            title="Hide conversations"
          >
            <PanelLeftClose className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>

      {/* New Chat Button */}
      <div className="px-2 pt-2 pb-1 shrink-0">
        <button
          type="button"
          onClick={onNewChat}
          className="w-full flex items-center gap-2 rounded-lg border border-dashed border-border px-2.5 py-2 text-[12px] text-muted-foreground hover:text-foreground hover:border-foreground/30 hover:bg-white/70 transition-colors"
        >
          <MessageSquarePlus className="h-3.5 w-3.5 shrink-0" />
          New Chat
        </button>
      </div>

      {/* Search */}
      <div className="px-2 pb-2 shrink-0">
        <div className="relative">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground/60" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => onSearchChange(e.target.value)}
            placeholder="Search chats..."
            className="w-full h-8 pl-8 pr-7 text-[12px] rounded-lg border border-border bg-white placeholder:text-muted-foreground/50 focus:outline-none focus:ring-1 focus:ring-foreground/10 focus:border-foreground/20"
          />
          {searchQuery && (
            <button
              onClick={() => onSearchChange("")}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground/60 hover:text-foreground"
            >
              <X className="h-3 w-3" />
            </button>
          )}
        </div>
      </div>

      {/* Conversation List */}
      <div className="flex-1 overflow-y-auto">
        {conversations.length === 0 && (
          <div className="flex flex-col items-center justify-center py-10 px-4 text-center">
            <MessageSquare className="h-8 w-8 text-muted-foreground/20 mb-2" />
            <p className="text-[11px] text-muted-foreground">No conversations yet</p>
            <p className="text-[10px] text-muted-foreground/60 mt-0.5">
              Click "New Chat" to start one
            </p>
          </div>
        )}

        {conversations.length > 0 && filtered.length === 0 && (
          <div className="flex flex-col items-center justify-center py-10 px-4 text-center">
            <Search className="h-6 w-6 text-muted-foreground/20 mb-2" />
            <p className="text-[11px] text-muted-foreground">No matches found</p>
          </div>
        )}

        {grouped.map((group) => (
          <div key={group.label}>
            <div className="px-3 pt-3 pb-1">
              <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground/50">
                {group.label}
              </span>
            </div>
            {group.items.map((conv) => {
              const displayTitle =
                conv.title ||
                (conv.last_message_preview
                  ? conv.last_message_preview.length > 45
                    ? conv.last_message_preview.slice(0, 45) + "..."
                    : conv.last_message_preview
                  : "Untitled conversation")

              const isActive = activeConversationId === conv.id

              return (
                <div
                  key={conv.id}
                  className={cn(
                    "group/item mx-1.5 mb-0.5 rounded-lg px-2.5 py-2 cursor-pointer transition-colors relative",
                    isActive
                      ? "bg-white border border-border shadow-sm"
                      : "hover:bg-white/70"
                  )}
                  onClick={() => {
                    setDeleteConfirmId(null)
                    onSelect(conv.id)
                  }}
                >
                  <p className="text-[12px] font-medium text-foreground leading-snug line-clamp-2 pr-1">
                    {highlightMatch(displayTitle, searchQuery)}
                  </p>
                  <div className="flex items-center justify-between mt-1">
                    <span className="text-[10px] text-muted-foreground/60">
                      {conv.message_count > 0 && `${conv.message_count} msg${conv.message_count !== 1 ? "s" : ""} · `}
                      {formatRelativeShort(conv.updated_at)}
                    </span>

                    {/* Delete control */}
                    {deleteConfirmId !== conv.id && (
                      <button
                        type="button"
                        className="opacity-0 group-hover/item:opacity-100 transition-opacity p-0.5 rounded hover:bg-red-50 text-muted-foreground/40 hover:text-red-500"
                        onClick={(e) => {
                          e.stopPropagation()
                          setDeleteConfirmId(conv.id)
                        }}
                        title="Delete conversation"
                      >
                        <Trash2 className="h-3 w-3" />
                      </button>
                    )}
                  </div>

                  <AnimatePresence>
                    {deleteConfirmId === conv.id && (
                      <motion.div
                        initial={{ opacity: 0, y: -4 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -4 }}
                        className="flex items-center gap-1 mt-1.5"
                        onClick={(e) => e.stopPropagation()}
                      >
                        <span className="text-[10px] text-red-600 flex-1">Delete?</span>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-5 text-[10px] px-1.5"
                          onClick={() => setDeleteConfirmId(null)}
                        >
                          No
                        </Button>
                        <Button
                          size="sm"
                          className="h-5 text-[10px] px-1.5 bg-red-600 hover:bg-red-700 text-white"
                          onClick={() => {
                            onDelete(conv.id)
                            setDeleteConfirmId(null)
                          }}
                          disabled={isDeleting}
                        >
                          {isDeleting ? <Loader2 className="h-3 w-3 animate-spin" /> : "Yes"}
                        </Button>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              )
            })}
          </div>
        ))}
      </div>
    </div>
  )
}
