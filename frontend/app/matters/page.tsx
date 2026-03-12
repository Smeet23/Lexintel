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
  Square,
} from "lucide-react"
import AppLayout from "@/layouts/AppLayout"
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
  if (status === "cancelled") return "Cancelled"
  return status
}

export default function MattersPage() {
  const router = useRouter()
  const [search, setSearch] = useState("")
  const [statusFilter, setStatusFilter] = useState<string>("all")
  const [showNewDialog, setShowNewDialog] = useState(false)
  const [newTitle, setNewTitle] = useState("")
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])
  const fileInputRef = useRef<HTMLInputElement>(null)
  const uploadAbortRef = useRef<AbortController | null>(null)

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
    if (!newTitle.trim() || selectedFiles.length === 0) return

    const controller = new AbortController()
    uploadAbortRef.current = controller

    try {
      const result = await createMatter.mutateAsync({
        name: newTitle.trim(),
        files: selectedFiles,
        signal: controller.signal,
      })
      uploadAbortRef.current = null
      setShowNewDialog(false)
      setNewTitle("")
      setSelectedFiles([])
      router.push(`/matters/${result.id}`)
    } catch {
      // Error is available via createMatter.error; abort is expected when stopping
      uploadAbortRef.current = null
    }
  }

  const handleStopUpload = () => {
    uploadAbortRef.current?.abort()
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
            <p className="text-sm text-muted mt-0.5">{item.file_type.toUpperCase()}</p>
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
        <span className="text-muted text-sm">{item.updated_at ? formatRelativeTime(item.updated_at) : formatRelativeTime(item.created_at)}</span>
      ),
    },
    {
      key: "actions",
      header: "",
      className: "w-12",
      render: () => (
        <Button variant="ghost" size="icon" className="h-8 w-8" onClick={(e) => e.stopPropagation()}>
          <MoreHorizontal className="h-4 w-4 text-muted" />
        </Button>
      ),
    },
  ]

  return (
    <AppLayout title="Matters">
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

      {/* Filters */}
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.1 }}
        className="flex items-center gap-4 mb-8"
      >
        <div className="relative flex-1 max-w-sm">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted" />
          <Input
            placeholder="Search matters..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="pl-9 rounded-lg"
          />
        </div>
        <Select value={statusFilter} onValueChange={setStatusFilter}>
          <SelectTrigger className="w-40 rounded-lg">
            <Filter className="h-4 w-4 mr-2 text-muted" />
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

      {/* Table */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.15 }}
        className="bg-white rounded-xl border border-border shadow-elevated"
      >
        {isLoading ? (
          <div className="flex items-center justify-center py-16">
            <Loader2 className="h-5 w-5 animate-spin text-muted" />
            <span className="ml-2 text-sm text-muted">Loading matters...</span>
          </div>
        ) : error ? (
          <div className="py-16 text-center">
            <p className="text-sm text-muted">Failed to load matters. Is the backend running?</p>
          </div>
        ) : (
          <>
            <DataTable
              columns={columns}
              data={filtered}
              onRowClick={(item) => router.push(`/matters/${item.id}`)}
              emptyMessage="No matters found"
            />
            {filtered.length > 0 && (
              <div className="border-t border-border px-5 py-4 text-sm text-muted">
                Showing {filtered.length} of {allMatters.length} matters
              </div>
            )}
          </>
        )}
      </motion.div>

      {/* New Matter Dialog */}
      <Dialog
        open={showNewDialog}
        onOpenChange={(open) => {
          if (!open && createMatter.isPending) handleStopUpload()
          setShowNewDialog(open)
          if (!open) {
            setNewTitle("")
            setSelectedFiles([])
          }
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Create New Matter</DialogTitle>
            <DialogDescription>
              Upload one or more documents to begin analyzing. Supports PDF, DOCX, and TXT files.
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
              <label className="block text-sm font-medium text-foreground mb-2">Documents</label>
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.docx,.txt"
                multiple
                className="hidden"
                onChange={(e) => {
                  const newFiles = Array.from(e.target.files || [])
                  if (newFiles.length > 0) {
                    setSelectedFiles((prev) => [...prev, ...newFiles])
                  }
                  e.target.value = "" // reset so same file can be re-added
                }}
              />
              {selectedFiles.length > 0 ? (
                <div className="space-y-2">
                  {selectedFiles.map((file, idx) => (
                    <div key={`${file.name}-${idx}`} className="flex items-center gap-3 rounded-lg border border-border p-3">
                      <Briefcase className="h-4 w-4 text-muted shrink-0" />
                      <span className="text-sm text-foreground flex-1 truncate">{file.name}</span>
                      <span className="text-xs text-muted shrink-0">{file.type.split("/").pop()?.toUpperCase()}</span>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setSelectedFiles((prev) => prev.filter((_, i) => i !== idx))}
                      >
                        Remove
                      </Button>
                    </div>
                  ))}
                  <Button
                    variant="outline"
                    size="sm"
                    className="w-full justify-center gap-2 rounded-lg border-dashed"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <Plus className="h-4 w-4" />
                    Add More Files
                  </Button>
                </div>
              ) : (
                <Button
                  variant="outline"
                  className="w-full justify-center gap-2 rounded-lg border-dashed"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <Upload className="h-4 w-4" />
                  Choose Files
                </Button>
              )}
            </div>
          </div>
          <DialogFooter>
            {createMatter.isPending ? (
              <>
                <Button variant="outline" onClick={handleStopUpload} className="text-destructive border-destructive/50 hover:bg-destructive/10">
                  <Square className="h-4 w-4" />
                  Stop
                </Button>
                <Button disabled>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Uploading...
                </Button>
              </>
            ) : (
              <>
                <Button variant="outline" onClick={() => { setShowNewDialog(false); setNewTitle(""); setSelectedFiles([]) }}>
                  Cancel
                </Button>
                <Button
                  onClick={handleCreateMatter}
                  disabled={!newTitle.trim() || selectedFiles.length === 0}
                >
                  Create Matter
                </Button>
              </>
            )}
          </DialogFooter>
          {createMatter.isError &&
            (createMatter.error as { name?: string; code?: string })?.name !== "AbortError" &&
            (createMatter.error as { name?: string; code?: string })?.code !== "ERR_CANCELED" && (
            <p className="text-sm text-red-600 mt-2">
              Failed to create matter. Please try again.
            </p>
          )}
        </DialogContent>
      </Dialog>
    </AppLayout>
  )
}
