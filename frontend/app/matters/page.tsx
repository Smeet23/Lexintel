"use client"

import React, { useState, useRef } from "react"
import { useRouter } from "next/navigation"
import {
  Briefcase,
  Plus,
  Search,
  Filter,
  MoreHorizontal,
  Loader2,
  AlertCircle,
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
import { caseToMatter } from "@/lib/types"
import type { Matter } from "@/lib/types"
import { useCases, useCreateCase } from "@/hooks/use-cases"

export default function MattersPage() {
  const router = useRouter()
  const [search, setSearch] = useState("")
  const [statusFilter, setStatusFilter] = useState<string>("all")
  const [showNewDialog, setShowNewDialog] = useState(false)
  const [newTitle, setNewTitle] = useState("")
  const [newJurisdiction, setNewJurisdiction] = useState("")
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const { data: cases, isLoading, error } = useCases()
  const createCase = useCreateCase()

  const matters: Matter[] = (cases || []).map(caseToMatter)

  const filtered = matters.filter((m) => {
    const matchSearch = m.title.toLowerCase().includes(search.toLowerCase()) ||
      m.jurisdiction.toLowerCase().includes(search.toLowerCase())
    const matchStatus = statusFilter === "all" || m.status === statusFilter
    return matchSearch && matchStatus
  })

  const handleCreate = () => {
    if (!newTitle.trim() || !selectedFile) return

    createCase.mutate(
      { name: newTitle.trim(), file: selectedFile },
      {
        onSuccess: (data) => {
          setShowNewDialog(false)
          setNewTitle("")
          setNewJurisdiction("")
          setSelectedFile(null)
          router.push(`/matters/${data.id}`)
        },
      }
    )
  }

  const columns = [
    {
      key: "title",
      header: "Matter",
      render: (item: Matter) => (
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-accent/10">
            <Briefcase className="h-4 w-4 text-accent" />
          </div>
          <div>
            <p className="font-medium text-foreground">{item.title}</p>
            <p className="text-xs text-muted">{item.documentsCount} docs &middot; {item.queriesCount} queries</p>
          </div>
        </div>
      ),
    },
    {
      key: "jurisdiction",
      header: "Jurisdiction",
      render: (item: Matter) => <span className="text-foreground">{item.jurisdiction || "—"}</span>,
    },
    {
      key: "status",
      header: "Status",
      render: (item: Matter) => (
        <Badge variant={item.status as "active" | "review" | "closed"}>
          {item.status.charAt(0).toUpperCase() + item.status.slice(1)}
        </Badge>
      ),
    },
    {
      key: "lastActivity",
      header: "Last Activity",
      render: (item: Matter) => (
        <span className="text-muted">{formatRelativeTime(item.lastActivity)}</span>
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
      <div className="flex items-center gap-3 mb-6">
        <div className="relative flex-1 max-w-sm">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted" />
          <Input
            placeholder="Search matters..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="pl-9"
          />
        </div>
        <Select value={statusFilter} onValueChange={setStatusFilter}>
          <SelectTrigger className="w-40">
            <Filter className="h-4 w-4 mr-2 text-muted" />
            <SelectValue placeholder="Status" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Status</SelectItem>
            <SelectItem value="active">Active</SelectItem>
            <SelectItem value="review">In Review</SelectItem>
            <SelectItem value="closed">Closed</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Loading / Error States */}
      {isLoading && (
        <div className="flex items-center justify-center py-20">
          <Loader2 className="h-6 w-6 animate-spin text-accent" />
          <span className="ml-2 text-muted">Loading matters...</span>
        </div>
      )}

      {error && (
        <div className="flex items-center justify-center py-20 text-red-600">
          <AlertCircle className="h-5 w-5 mr-2" />
          <span>Failed to load matters. Is the backend running?</span>
        </div>
      )}

      {/* Table */}
      {!isLoading && !error && (
        <div className="bg-white rounded-xl border border-border shadow-sm">
          <DataTable
            columns={columns}
            data={filtered}
            onRowClick={(item) => router.push(`/matters/${item.id}`)}
            emptyMessage="No matters found"
          />
          {filtered.length > 0 && (
            <div className="border-t border-border px-4 py-3 text-xs text-muted">
              Showing {filtered.length} of {matters.length} matters
            </div>
          )}
        </div>
      )}

      {/* New Matter Dialog */}
      <Dialog open={showNewDialog} onOpenChange={setShowNewDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Create New Matter</DialogTitle>
            <DialogDescription>
              Set up a new legal matter by uploading a document for analysis.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 mt-4">
            <div>
              <label className="block text-sm font-medium text-foreground mb-1.5">Matter Title</label>
              <Input
                placeholder="e.g., Acme Corp Acquisition Review"
                value={newTitle}
                onChange={(e) => setNewTitle(e.target.value)}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-foreground mb-1.5">Jurisdiction</label>
              <Select value={newJurisdiction} onValueChange={setNewJurisdiction}>
                <SelectTrigger>
                  <SelectValue placeholder="Select jurisdiction" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="us-federal">US - Federal</SelectItem>
                  <SelectItem value="us-california">US - California</SelectItem>
                  <SelectItem value="us-new-york">US - New York</SelectItem>
                  <SelectItem value="us-delaware">US - Delaware</SelectItem>
                  <SelectItem value="uk">United Kingdom</SelectItem>
                  <SelectItem value="eu">European Union</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div>
              <label className="block text-sm font-medium text-foreground mb-1.5">Document</label>
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.docx,.txt"
                className="hidden"
                onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
              />
              <div
                className="border-2 border-dashed border-border rounded-lg p-4 text-center cursor-pointer hover:border-accent/50 transition-colors"
                onClick={() => fileInputRef.current?.click()}
              >
                {selectedFile ? (
                  <p className="text-sm text-foreground">{selectedFile.name}</p>
                ) : (
                  <p className="text-sm text-muted">Click to select a PDF, DOCX, or TXT file</p>
                )}
              </div>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowNewDialog(false)}>
              Cancel
            </Button>
            <Button
              onClick={handleCreate}
              disabled={!newTitle.trim() || !selectedFile || createCase.isPending}
            >
              {createCase.isPending && <Loader2 className="h-4 w-4 animate-spin" />}
              {createCase.isPending ? "Creating..." : "Create Matter"}
            </Button>
          </DialogFooter>
          {createCase.isError && (
            <p className="text-sm text-red-600 mt-2">
              Failed to create matter. Please try again.
            </p>
          )}
        </DialogContent>
      </Dialog>
    </AppLayout>
  )
}
