"use client"

import React, { useState } from "react"
import { useRouter } from "next/navigation"
import {
  Briefcase,
  Plus,
  Search,
  Filter,
  MoreHorizontal,
  FileText,
  Trash2,
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

interface Matter {
  id: string
  title: string
  jurisdiction: string
  status: "active" | "review" | "closed" | "archived"
  documentsCount: number
  queriesCount: number
  tokenUsage: number
  lastActivity: string
  createdAt: string
}

const mockMatters: Matter[] = [
  { id: "1", title: "Acme vs Global Corp", jurisdiction: "US - Federal", status: "active", documentsCount: 12, queriesCount: 45, tokenUsage: 8234, lastActivity: new Date(Date.now() - 7200000).toISOString(), createdAt: "2025-12-01" },
  { id: "2", title: "Smith Estate Planning", jurisdiction: "US - California", status: "active", documentsCount: 8, queriesCount: 23, tokenUsage: 4120, lastActivity: new Date(Date.now() - 18000000).toISOString(), createdAt: "2025-12-15" },
  { id: "3", title: "TechStart IP Review", jurisdiction: "US - Delaware", status: "review", documentsCount: 23, queriesCount: 67, tokenUsage: 12890, lastActivity: new Date(Date.now() - 86400000).toISOString(), createdAt: "2026-01-03" },
  { id: "4", title: "Metro Construction Dispute", jurisdiction: "UK", status: "active", documentsCount: 15, queriesCount: 31, tokenUsage: 6450, lastActivity: new Date(Date.now() - 172800000).toISOString(), createdAt: "2026-01-10" },
  { id: "5", title: "Phoenix Merger Analysis", jurisdiction: "EU", status: "closed", documentsCount: 34, queriesCount: 89, tokenUsage: 18900, lastActivity: new Date(Date.now() - 604800000).toISOString(), createdAt: "2025-11-20" },
  { id: "6", title: "Harbor Real Estate Lease", jurisdiction: "US - New York", status: "active", documentsCount: 6, queriesCount: 12, tokenUsage: 2340, lastActivity: new Date(Date.now() - 3600000).toISOString(), createdAt: "2026-01-28" },
  { id: "7", title: "Nordic Data Privacy Audit", jurisdiction: "EU - GDPR", status: "review", documentsCount: 18, queriesCount: 41, tokenUsage: 9870, lastActivity: new Date(Date.now() - 259200000).toISOString(), createdAt: "2026-01-15" },
  { id: "8", title: "Cypress Insurance Claim", jurisdiction: "US - Florida", status: "active", documentsCount: 9, queriesCount: 28, tokenUsage: 5620, lastActivity: new Date(Date.now() - 43200000).toISOString(), createdAt: "2026-02-01" },
]

export default function MattersPage() {
  const router = useRouter()
  const [search, setSearch] = useState("")
  const [statusFilter, setStatusFilter] = useState<string>("all")
  const [showNewDialog, setShowNewDialog] = useState(false)
  const [newTitle, setNewTitle] = useState("")
  const [newJurisdiction, setNewJurisdiction] = useState("")

  const filtered = mockMatters.filter((m) => {
    const matchSearch = m.title.toLowerCase().includes(search.toLowerCase()) ||
      m.jurisdiction.toLowerCase().includes(search.toLowerCase())
    const matchStatus = statusFilter === "all" || m.status === statusFilter
    return matchSearch && matchStatus
  })

  const columns = [
    {
      key: "title",
      header: "Matter",
      render: (item: Matter) => (
        <div className="flex items-center gap-4">
          <div className="flex h-10 w-10 items-center justify-center rounded-sm bg-surface">
            <Briefcase className="h-4 w-4 text-foreground" />
          </div>
          <div>
            <p className="font-medium text-foreground text-base">{item.title}</p>
            <p className="text-sm text-muted mt-0.5">{item.documentsCount} docs &middot; {item.queriesCount} queries</p>
          </div>
        </div>
      ),
    },
    {
      key: "jurisdiction",
      header: "Jurisdiction",
      render: (item: Matter) => <span className="text-foreground text-sm">{item.jurisdiction}</span>,
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
      key: "tokenUsage",
      header: "Tokens",
      render: (item: Matter) => (
        <span className="text-muted font-mono text-sm">{item.tokenUsage.toLocaleString()}</span>
      ),
    },
    {
      key: "lastActivity",
      header: "Last Activity",
      render: (item: Matter) => (
        <span className="text-muted text-sm">{formatRelativeTime(item.lastActivity)}</span>
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
      <div className="flex items-center gap-4 mb-8">
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

      {/* Table */}
      <div className="bg-white rounded-sm border border-border shadow-sm">
        <DataTable
          columns={columns}
          data={filtered}
          onRowClick={(item) => router.push(`/matters/${item.id}`)}
          emptyMessage="No matters found"
        />
        {filtered.length > 0 && (
          <div className="border-t border-border px-5 py-4 text-sm text-muted">
            Showing {filtered.length} of {mockMatters.length} matters
          </div>
        )}
      </div>

      {/* New Matter Dialog */}
      <Dialog open={showNewDialog} onOpenChange={setShowNewDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Create New Matter</DialogTitle>
            <DialogDescription>
              Set up a new legal matter to begin uploading documents and querying.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-5 mt-4">
            <div>
              <label className="block text-sm font-medium text-foreground mb-2">Matter Title</label>
              <Input
                placeholder="e.g., Acme Corp Acquisition Review"
                value={newTitle}
                onChange={(e) => setNewTitle(e.target.value)}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-foreground mb-2">Jurisdiction</label>
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
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowNewDialog(false)}>
              Cancel
            </Button>
            <Button onClick={() => { setShowNewDialog(false); router.push("/matters/new") }}>
              Create Matter
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </AppLayout>
  )
}
