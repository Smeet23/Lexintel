"use client"

import React, { useState, useRef } from "react"
import { useRouter } from "next/navigation"
import { motion } from "framer-motion"
import {
  Briefcase,
  Plus,
  Search,
  Filter,
  MoreHorizontal,
  Loader2,
  Upload,
} from "lucide-react"
import PageHeader from "@/components/PageHeader"
import DataTable from "@/components/DataTable"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { formatRelativeTime } from "@/lib/utils"
import { useMatters, useCreateMatter, useDeleteMatter } from "@/hooks/use-matters"
import type { MatterResponse } from "@/lib/api-services"

function mapStatus(status: string): "active" | "review" | "closed" {
  if (status === "ready") return "active"
  if (status === "processing") return "review"
  return "closed"
}

function statusLabel(status: string): string {
  if (status === "ready") return "Ready"
  if (status === "processing") return "Processing"
  if (status === "error") return "Error"
  return status
}

export default function MattersView({ firmSlug }: { firmSlug?: string }) {
  const router = useRouter()
  const prefix = firmSlug ? `/firm/${firmSlug}` : ""
  const [search, setSearch] = useState("")
  const [statusFilter, setStatusFilter] = useState<string>("all")
  const [showNewDialog, setShowNewDialog] = useState(false)
  const [newTitle, setNewTitle] = useState("")
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const { data: matters, isLoading, error } = useMatters()
  const createMatter = useCreateMatter()
  const deleteMatter = useDeleteMatter()

  const allMatters = matters || []

  const filtered = allMatters.filter((m) => {
    const matchSearch = m.name.toLowerCase().includes(search.toLowerCase())
    const matchStatus = statusFilter === "all" || m.status === statusFilter
    return matchSearch && matchStatus
  })

  const handleCreateMatter = async () => {
    if (!newTitle.trim() || !selectedFile) return

    try {
      const result = await createMatter.mutateAsync({ name: newTitle.trim(), file: selectedFile })
      setShowNewDialog(false)
      setNewTitle("")
      setSelectedFile(null)
      router.push(`${prefix}/matters/${result.id}`)
    } catch {
      // Error is available via createMatter.error
    }
  }

  const columns = [
    {
      key: "title",
      header: "Matter",
      render: (item: MatterResponse) => (
        <div className="flex items-center gap-4">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-surface">
            <Briefcase className="h-4 w-4 text-foreground" />
          </div>
          <div>
            <p className="font-medium text-foreground text-base">{item.name}</p>
            <p className="text-sm text-muted-foreground mt-0.5">{item.file_type.toUpperCase()}</p>
          </div>
        </div>
      ),
    },
    {
      key: "status",
      header: "Status",
      render: (item: MatterResponse) => (
        <Badge variant={mapStatus(item.status)}>
          {statusLabel(item.status)}
        </Badge>
      ),
    },
    {
      key: "lastActivity",
      header: "Last Activity",
      render: (item: MatterResponse) => (
        <span className="text-muted-foreground text-sm">{item.updated_at ? formatRelativeTime(item.updated_at) : formatRelativeTime(item.created_at)}</span>
      ),
    },
    {
      key: "actions",
      header: "",
      className: "w-12",
      render: () => (
        <Button variant="ghost" size="icon" className="h-8 w-8" onClick={(e) => e.stopPropagation()}>
          <MoreHorizontal className="h-4 w-4 text-muted-foreground" />
        </Button>
      ),
    },
  ]

  return (
    <>
      <PageHeader
        title="Matters"
        description="Manage your legal matters and case files"
        actions={
          <Button onClick={() => setShowNewDialog(true)}>
            <Plus className="h-4 w-4" />
            New Matter
          </Button>
        }
      />

      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.1 }}
        className="flex items-center gap-4 mb-8"
      >
        <div className="relative flex-1 max-w-sm">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="Search matters..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="pl-9 rounded-lg"
          />
        </div>
        <Select value={statusFilter} onValueChange={setStatusFilter}>
          <SelectTrigger className="w-40 rounded-lg">
            <Filter className="h-4 w-4 mr-2 text-muted-foreground" />
            <SelectValue placeholder="Status" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Status</SelectItem>
            <SelectItem value="ready">Ready</SelectItem>
            <SelectItem value="processing">Processing</SelectItem>
            <SelectItem value="error">Error</SelectItem>
          </SelectContent>
        </Select>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.15 }}
        className="bg-white rounded-xl border border-border shadow-elevated"
      >
        {isLoading ? (
          <div className="flex items-center justify-center py-16">
            <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
            <span className="ml-2 text-sm text-muted-foreground">Loading matters...</span>
          </div>
        ) : error ? (
          <div className="py-16 text-center">
            <p className="text-sm text-muted-foreground">Failed to load matters. Is the backend running?</p>
          </div>
        ) : (
          <>
            <DataTable
              columns={columns}
              data={filtered}
              onRowClick={(item) => router.push(`${prefix}/matters/${item.id}`)}
              emptyMessage="No matters found"
            />
            {filtered.length > 0 && (
              <div className="border-t border-border px-5 py-4 text-sm text-muted-foreground">
                Showing {filtered.length} of {allMatters.length} matters
              </div>
            )}
          </>
        )}
      </motion.div>

      <Dialog open={showNewDialog} onOpenChange={setShowNewDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Create New Matter</DialogTitle>
            <DialogDescription>
              Upload a document to begin analyzing. Supports PDF, DOCX, and TXT files.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-5 mt-4">
            <div>
              <label className="block text-sm font-medium text-foreground mb-2">Matter Name</label>
              <Input
                placeholder="e.g., Acme Corp Acquisition Review"
                value={newTitle}
                onChange={(e) => setNewTitle(e.target.value)}
                className="rounded-lg"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-foreground mb-2">Document</label>
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.docx,.txt"
                className="hidden"
                onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
              />
              {selectedFile ? (
                <div className="flex items-center gap-3 rounded-lg border border-border p-3">
                  <Briefcase className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm text-foreground flex-1 truncate">{selectedFile.name}</span>
                  <Button variant="ghost" size="sm" onClick={() => setSelectedFile(null)}>
                    Remove
                  </Button>
                </div>
              ) : (
                <Button
                  variant="outline"
                  className="w-full justify-center gap-2 rounded-lg border-dashed"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <Upload className="h-4 w-4" />
                  Choose File
                </Button>
              )}
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => { setShowNewDialog(false); setNewTitle(""); setSelectedFile(null) }}>
              Cancel
            </Button>
            <Button
              onClick={handleCreateMatter}
              disabled={!newTitle.trim() || !selectedFile || createMatter.isPending}
            >
              {createMatter.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Creating...
                </>
              ) : (
                "Create Matter"
              )}
            </Button>
          </DialogFooter>
          {createMatter.isError && (
            <p className="text-sm text-red-600 mt-2">
              Failed to create matter. Please try again.
            </p>
          )}
        </DialogContent>
      </Dialog>
    </>
  )
}
