"use client"

import React, { useState } from "react"
import { BookOpen, Search, Tag, Calendar, Briefcase, Star, Filter, Plus } from "lucide-react"
import AppLayout from "@/layouts/AppLayout"
import PageHeader from "@/components/PageHeader"
import DataTable from "@/components/DataTable"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { formatDate } from "@/lib/utils"

interface Precedent {
  id: string
  title: string
  jurisdiction: string
  practiceArea: string
  matterTitle: string
  tags: string[]
  savedBy: string
  savedAt: string
  [key: string]: unknown
}

const mockPrecedents: Precedent[] = [
  { id: "p1", title: "Indemnification Clause Analysis - Standard Commercial", jurisdiction: "US - Federal", practiceArea: "Commercial", matterTitle: "Acme vs Global Corp", tags: ["indemnification", "commercial", "risk"], savedBy: "John Smith", savedAt: "2026-01-15" },
  { id: "p2", title: "GDPR Data Processing Agreement Template Review", jurisdiction: "EU", practiceArea: "Data Privacy", matterTitle: "Nordic Data Privacy Audit", tags: ["gdpr", "dpa", "privacy"], savedBy: "Sarah Chen", savedAt: "2026-01-20" },
  { id: "p3", title: "IP Assignment Provisions - Tech Companies", jurisdiction: "US - Delaware", practiceArea: "Intellectual Property", matterTitle: "TechStart IP Review", tags: ["ip", "assignment", "tech"], savedBy: "John Smith", savedAt: "2026-01-08" },
  { id: "p4", title: "Real Estate Lease Termination - Commercial Property", jurisdiction: "US - New York", practiceArea: "Real Estate", matterTitle: "Harbor Real Estate Lease", tags: ["lease", "termination", "real-estate"], savedBy: "Lisa Park", savedAt: "2026-02-01" },
  { id: "p5", title: "Insurance Coverage Dispute Analysis", jurisdiction: "US - Florida", practiceArea: "Insurance", matterTitle: "Cypress Insurance Claim", tags: ["insurance", "coverage", "dispute"], savedBy: "Mike Torres", savedAt: "2026-02-05" },
  { id: "p6", title: "Merger Anti-Trust Review Framework", jurisdiction: "EU", practiceArea: "M&A", matterTitle: "Phoenix Merger Analysis", tags: ["merger", "antitrust", "eu"], savedBy: "Sarah Chen", savedAt: "2025-12-10" },
]

export default function PrecedentsPage() {
  const [search, setSearch] = useState("")
  const [areaFilter, setAreaFilter] = useState("all")

  const filtered = mockPrecedents.filter((p) => {
    const matchSearch = p.title.toLowerCase().includes(search.toLowerCase()) ||
      p.tags.some((t) => t.includes(search.toLowerCase()))
    const matchArea = areaFilter === "all" || p.practiceArea === areaFilter
    return matchSearch && matchArea
  })

  const columns = [
    {
      key: "title",
      header: "Precedent",
      render: (item: Precedent) => (
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-surface">
            <BookOpen className="h-4 w-4 text-foreground" />
          </div>
          <div>
            <p className="font-medium text-foreground text-sm">{item.title}</p>
            <p className="text-xs text-muted mt-0.5">
              <Briefcase className="inline h-3 w-3 mr-1" />{item.matterTitle}
            </p>
          </div>
        </div>
      ),
    },
    {
      key: "jurisdiction",
      header: "Jurisdiction",
      render: (item: Precedent) => <span className="text-sm">{item.jurisdiction}</span>,
    },
    {
      key: "practiceArea",
      header: "Practice Area",
      render: (item: Precedent) => <Badge variant="default">{item.practiceArea}</Badge>,
    },
    {
      key: "tags",
      header: "Tags",
      render: (item: Precedent) => (
        <div className="flex flex-wrap gap-1">
          {item.tags.slice(0, 3).map((tag) => (
            <span key={tag} className="inline-flex items-center rounded-lg bg-surface px-2 py-0.5 text-xs text-muted">
              {tag}
            </span>
          ))}
        </div>
      ),
    },
    {
      key: "savedAt",
      header: "Saved",
      render: (item: Precedent) => (
        <div className="text-sm">
          <p className="text-muted">{formatDate(item.savedAt)}</p>
          <p className="text-xs text-muted/70">{item.savedBy}</p>
        </div>
      ),
    },
  ]

  const practiceAreas = [...new Set(mockPrecedents.map((p) => p.practiceArea))]

  return (
    <AppLayout title="Precedents">
      <PageHeader
        title="Precedent Library"
        description="Firm-wide approved outputs and research saved across matters"
        actions={
          <Button>
            <Plus className="h-4 w-4" />
            Save New Precedent
          </Button>
        }
      />

      {/* Filters */}
      <div className="flex items-center gap-3 mb-8">
        <div className="relative flex-1 max-w-sm">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted" />
          <Input
            placeholder="Search precedents or tags..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="pl-9"
          />
        </div>
        <Select value={areaFilter} onValueChange={setAreaFilter}>
          <SelectTrigger className="w-48">
            <Filter className="h-4 w-4 mr-2 text-muted" />
            <SelectValue placeholder="Practice Area" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Practice Areas</SelectItem>
            {practiceAreas.map((area) => (
              <SelectItem key={area} value={area}>{area}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Table */}
      <div className="bg-white rounded-xl border border-border shadow-elevated">
        <DataTable
          columns={columns}
          data={filtered}
          emptyMessage="No precedents found"
        />
        {filtered.length > 0 && (
          <div className="border-t border-border px-4 py-3 text-xs text-muted">
            {filtered.length} precedent{filtered.length !== 1 ? "s" : ""}
          </div>
        )}
      </div>
    </AppLayout>
  )
}
