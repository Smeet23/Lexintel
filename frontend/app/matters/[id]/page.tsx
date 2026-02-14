"use client"

import React, { useState, useCallback } from "react"
import { useParams, useRouter } from "next/navigation"
import {
  ArrowLeft,
  MessageSquare,
  FileText,
  Shield,
  PenLine,
  ClipboardList,
  Upload,
  Loader2,
  CheckCircle2,
  AlertTriangle,
  Clock,
  Download,
  Eye,
  Trash2,
  ChevronRight,
} from "lucide-react"
import AppLayout from "@/layouts/AppLayout"
import ChatPanel from "@/components/ChatPanel"
import CitationPanel from "@/components/CitationPanel"
import DataTable from "@/components/DataTable"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Progress } from "@/components/ui/progress"
import { cn, formatRelativeTime } from "@/lib/utils"
import type { QueryMessage, Citation, Document, AuditEntry } from "@/lib/types"

// Mock data
const matterData = {
  id: "1",
  title: "Acme vs Global Corp",
  jurisdiction: "US - Federal",
  status: "active" as const,
  team: ["John Smith", "Sarah Chen"],
  documentsCount: 12,
  queriesCount: 45,
  tokenUsage: 8234,
  budget: 15000,
  createdAt: "2025-12-01",
}

const mockDocuments: (Document & Record<string, unknown>)[] = [
  { id: "d1", matterId: "1", name: "Master Services Agreement.pdf", fileType: "pdf", status: "indexed", pageCount: 42, size: 2400000, uploadedAt: new Date(Date.now() - 86400000).toISOString(), uploadedBy: "John Smith" },
  { id: "d2", matterId: "1", name: "Amendment No. 3.docx", fileType: "docx", status: "indexed", pageCount: 8, size: 340000, uploadedAt: new Date(Date.now() - 172800000).toISOString(), uploadedBy: "Sarah Chen" },
  { id: "d3", matterId: "1", name: "Exhibit A - Financial Statements.pdf", fileType: "pdf", status: "processing", pageCount: 156, size: 8900000, uploadedAt: new Date(Date.now() - 3600000).toISOString(), uploadedBy: "John Smith" },
  { id: "d4", matterId: "1", name: "Counterparty Due Diligence Report.pdf", fileType: "pdf", status: "indexed", pageCount: 34, size: 1800000, uploadedAt: new Date(Date.now() - 259200000).toISOString(), uploadedBy: "Lisa Park" },
  { id: "d5", matterId: "1", name: "Board Resolution.txt", fileType: "txt", status: "indexed", size: 12000, uploadedAt: new Date(Date.now() - 604800000).toISOString(), uploadedBy: "John Smith" },
]

const mockAuditLog: (AuditEntry & Record<string, unknown>)[] = [
  { id: "a1", action: "Query", user: "John Smith", details: "Summarize key risks in this contract", sources: ["MSA.pdf - Page 12", "Amendment 3 - Page 4"], timestamp: new Date(Date.now() - 3600000).toISOString() },
  { id: "a2", action: "Upload", user: "John Smith", details: "Uploaded Exhibit A - Financial Statements.pdf", timestamp: new Date(Date.now() - 7200000).toISOString() },
  { id: "a3", action: "Query", user: "Sarah Chen", details: "What are the termination provisions?", sources: ["MSA.pdf - Page 28", "MSA.pdf - Page 29"], timestamp: new Date(Date.now() - 86400000).toISOString() },
  { id: "a4", action: "Export", user: "John Smith", details: "Exported evidence bundle for indemnification analysis", timestamp: new Date(Date.now() - 172800000).toISOString() },
  { id: "a5", action: "Upload", user: "Sarah Chen", details: "Uploaded Amendment No. 3.docx", timestamp: new Date(Date.now() - 259200000).toISOString() },
]

// Contract review mock
const contractRisks = [
  { clause: "Indemnification (Section 8.2)", risk: "high" as const, summary: "Unlimited indemnification exposure for IP infringement claims. No cap or basket provisions." },
  { clause: "Termination for Convenience (Section 12.1)", risk: "medium" as const, summary: "30-day notice period may be insufficient. No wind-down provisions for ongoing work." },
  { clause: "Limitation of Liability (Section 9.1)", risk: "low" as const, summary: "Standard mutual cap at 12 months' fees. Carve-outs are appropriate." },
  { clause: "Data Protection (Section 14)", risk: "high" as const, summary: "Missing sub-processor notification requirements. GDPR Article 28 compliance gap." },
  { clause: "Assignment (Section 16.3)", risk: "medium" as const, summary: "Silent on change of control assignment. May allow unwanted counterparty changes." },
]

export default function MatterWorkspacePage() {
  const params = useParams()
  const router = useRouter()
  const [messages, setMessages] = useState<QueryMessage[]>([])
  const [selectedCitations, setSelectedCitations] = useState<Citation[]>([])
  const [isQuerying, setIsQuerying] = useState(false)
  const [draftType, setDraftType] = useState("")

  const handleSendMessage = useCallback((content: string) => {
    const userMsg: QueryMessage = {
      id: `msg-${Date.now()}`,
      role: "user",
      content,
      timestamp: new Date().toISOString(),
    }
    setMessages((prev) => [...prev, userMsg])
    setIsQuerying(true)

    // Simulate AI response
    setTimeout(() => {
      const aiMsg: QueryMessage = {
        id: `msg-${Date.now() + 1}`,
        role: "assistant",
        content: `Based on my analysis of the uploaded documents, here is my response to your question:\n\nThe Master Services Agreement contains several key provisions relevant to your query. Specifically, Section 8.2 outlines indemnification obligations with unlimited exposure for IP claims, while Section 12.1 provides for termination with a 30-day notice period.\n\nI identified 3 documents containing relevant information, with the strongest authority found in the MSA (pages 12-15) and Amendment No. 3 (pages 4-5).`,
        citations: [
          { documentName: "Master Services Agreement.pdf", pageNumber: 12, section: "Section 8.2 - Indemnification", excerpt: "Each party shall indemnify and hold harmless the other party from any claims arising from...", relevanceScore: 0.94 },
          { documentName: "Master Services Agreement.pdf", pageNumber: 28, section: "Section 12.1 - Termination", excerpt: "Either party may terminate this Agreement upon thirty (30) days written notice...", relevanceScore: 0.87 },
          { documentName: "Amendment No. 3.docx", pageNumber: 4, section: "Amendment to Section 8", excerpt: "The indemnification provisions of the Agreement are hereby amended to include...", relevanceScore: 0.82 },
        ],
        confidenceScore: 87,
        timestamp: new Date().toISOString(),
      }
      setMessages((prev) => [...prev, aiMsg])
      setSelectedCitations(aiMsg.citations || [])
      setIsQuerying(false)
    }, 2500)
  }, [])

  const docColumns = [
    {
      key: "name",
      header: "Document",
      render: (item: typeof mockDocuments[0]) => (
        <div className="flex items-center gap-3">
          <div className={cn(
            "flex h-9 w-9 items-center justify-center rounded-lg",
            item.fileType === "pdf" ? "bg-red-50" : item.fileType === "docx" ? "bg-blue-50" : "bg-slate-50"
          )}>
            <FileText className={cn(
              "h-4 w-4",
              item.fileType === "pdf" ? "text-red-500" : item.fileType === "docx" ? "text-blue-500" : "text-slate-500"
            )} />
          </div>
          <div>
            <p className="font-medium text-foreground text-sm">{item.name}</p>
            <p className="text-xs text-muted">
              {item.pageCount ? `${item.pageCount} pages` : ""} &middot; {(item.size / 1024 / 1024).toFixed(1)} MB
            </p>
          </div>
        </div>
      ),
    },
    {
      key: "status",
      header: "Status",
      render: (item: typeof mockDocuments[0]) => (
        <Badge variant={item.status as "indexed" | "processing" | "error"}>
          {item.status === "indexed" && <CheckCircle2 className="h-3 w-3 mr-1" />}
          {item.status === "processing" && <Loader2 className="h-3 w-3 mr-1 animate-spin" />}
          {item.status.charAt(0).toUpperCase() + item.status.slice(1)}
        </Badge>
      ),
    },
    {
      key: "uploadedBy",
      header: "Uploaded By",
      render: (item: typeof mockDocuments[0]) => <span className="text-muted text-sm">{item.uploadedBy as string}</span>,
    },
    {
      key: "uploadedAt",
      header: "Date",
      render: (item: typeof mockDocuments[0]) => <span className="text-muted text-sm">{formatRelativeTime(item.uploadedAt)}</span>,
    },
    {
      key: "actions",
      header: "",
      className: "w-24",
      render: () => (
        <div className="flex gap-1">
          <Button variant="ghost" size="icon" className="h-8 w-8"><Eye className="h-4 w-4 text-muted" /></Button>
          <Button variant="ghost" size="icon" className="h-8 w-8"><Trash2 className="h-4 w-4 text-muted" /></Button>
        </div>
      ),
    },
  ]

  const auditColumns = [
    {
      key: "action",
      header: "Action",
      render: (item: typeof mockAuditLog[0]) => (
        <Badge variant={item.action === "Query" ? "default" : item.action === "Upload" ? "active" : "review"}>
          {item.action as string}
        </Badge>
      ),
    },
    {
      key: "user",
      header: "User",
      render: (item: typeof mockAuditLog[0]) => <span className="text-sm font-medium">{item.user}</span>,
    },
    {
      key: "details",
      header: "Details",
      render: (item: typeof mockAuditLog[0]) => <span className="text-sm text-muted">{item.details}</span>,
    },
    {
      key: "sources",
      header: "Sources",
      render: (item: typeof mockAuditLog[0]) => (
        <span className="text-xs text-muted">
          {(item.sources as string[] | undefined)?.join(", ") || "-"}
        </span>
      ),
    },
    {
      key: "timestamp",
      header: "Time",
      render: (item: typeof mockAuditLog[0]) => <span className="text-xs text-muted">{formatRelativeTime(item.timestamp)}</span>,
    },
  ]

  return (
    <AppLayout title={matterData.title}>
      {/* Breadcrumb + Header */}
      <div className="mb-6">
        <button
          onClick={() => router.push("/matters")}
          className="flex items-center gap-1 text-sm text-muted hover:text-foreground transition-colors mb-3 cursor-pointer"
        >
          <ArrowLeft className="h-4 w-4" /> Back to Matters
        </button>

        <div className="flex items-start justify-between">
          <div>
            <div className="flex items-center gap-3">
              <h2 className="text-2xl font-semibold text-foreground">{matterData.title}</h2>
              <Badge variant="active">Active</Badge>
            </div>
            <p className="text-sm text-muted mt-1">
              {matterData.jurisdiction} &middot; {matterData.documentsCount} documents &middot; Created {matterData.createdAt}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm">
              <Download className="h-4 w-4" />
              Export Bundle
            </Button>
            <Button size="sm">
              <Upload className="h-4 w-4" />
              Upload Document
            </Button>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <Tabs defaultValue="ask-ai" className="space-y-4">
        <TabsList>
          <TabsTrigger value="ask-ai">
            <MessageSquare className="h-4 w-4 mr-2" />
            Ask AI
          </TabsTrigger>
          <TabsTrigger value="documents">
            <FileText className="h-4 w-4 mr-2" />
            Documents
          </TabsTrigger>
          <TabsTrigger value="contract-review">
            <Shield className="h-4 w-4 mr-2" />
            Contract Review
          </TabsTrigger>
          <TabsTrigger value="draft-assistant">
            <PenLine className="h-4 w-4 mr-2" />
            Draft Assistant
          </TabsTrigger>
          <TabsTrigger value="audit-log">
            <ClipboardList className="h-4 w-4 mr-2" />
            Audit Log
          </TabsTrigger>
        </TabsList>

        {/* Ask AI Tab */}
        <TabsContent value="ask-ai">
          <div className="flex gap-6 h-[calc(100vh-280px)]">
            <div className="flex-1 bg-white rounded-xl border border-border shadow-sm overflow-hidden">
              <ChatPanel
                messages={messages}
                onSend={handleSendMessage}
                isLoading={isQuerying}
                onSelectCitation={setSelectedCitations}
              />
            </div>
            <div className="w-80 bg-white rounded-xl border border-border shadow-sm p-5 overflow-y-auto">
              <CitationPanel citations={selectedCitations} />
            </div>
          </div>
        </TabsContent>

        {/* Documents Tab */}
        <TabsContent value="documents">
          <div className="space-y-4">
            {/* Upload zone */}
            <div className="border-2 border-dashed border-border rounded-xl p-8 text-center hover:border-accent/50 transition-colors bg-white">
              <Upload className="h-8 w-8 text-muted mx-auto mb-3" />
              <p className="text-sm font-medium text-foreground">
                Drag & drop files here, or click to browse
              </p>
              <p className="text-xs text-muted mt-1">
                Supports PDF, DOCX, and TXT files up to 50MB
              </p>
              <Button variant="outline" size="sm" className="mt-4">
                Browse Files
              </Button>
            </div>

            {/* Documents list */}
            <div className="bg-white rounded-xl border border-border shadow-sm">
              <div className="p-4 border-b border-border flex items-center justify-between">
                <h3 className="font-semibold text-foreground">Documents ({mockDocuments.length})</h3>
                <div className="flex items-center gap-2">
                  <Input placeholder="Search documents..." className="w-48 h-8 text-sm" />
                </div>
              </div>
              <DataTable columns={docColumns} data={mockDocuments} />
            </div>
          </div>
        </TabsContent>

        {/* Contract Review Tab */}
        <TabsContent value="contract-review">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 space-y-4">
              <div className="bg-white rounded-xl border border-border shadow-sm p-6">
                <h3 className="text-lg font-semibold text-foreground mb-4">Risk Analysis</h3>
                <div className="space-y-3">
                  {contractRisks.map((risk, idx) => (
                    <div
                      key={idx}
                      className={cn(
                        "rounded-lg border p-4 transition-colors hover:shadow-sm cursor-pointer",
                        risk.risk === "high" ? "border-red-200 bg-red-50/50" :
                        risk.risk === "medium" ? "border-amber-200 bg-amber-50/50" :
                        "border-emerald-200 bg-emerald-50/50"
                      )}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex items-start gap-3">
                          <AlertTriangle className={cn(
                            "h-5 w-5 mt-0.5 shrink-0",
                            risk.risk === "high" ? "text-red-500" :
                            risk.risk === "medium" ? "text-amber-500" :
                            "text-emerald-500"
                          )} />
                          <div>
                            <p className="font-medium text-foreground text-sm">{risk.clause}</p>
                            <p className="text-sm text-muted mt-1">{risk.summary}</p>
                          </div>
                        </div>
                        <Badge variant={risk.risk === "high" ? "error" : risk.risk === "medium" ? "review" : "active"}>
                          {risk.risk.charAt(0).toUpperCase() + risk.risk.slice(1)} Risk
                        </Badge>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="space-y-4">
              <div className="bg-white rounded-xl border border-border shadow-sm p-6">
                <h4 className="font-semibold text-foreground mb-4">Summary</h4>
                <div className="space-y-3">
                  <div className="flex justify-between text-sm">
                    <span className="text-muted">Total Clauses Analyzed</span>
                    <span className="font-medium">24</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-muted">High Risk</span>
                    <span className="font-medium text-red-600">2</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-muted">Medium Risk</span>
                    <span className="font-medium text-amber-600">2</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-muted">Low Risk</span>
                    <span className="font-medium text-emerald-600">1</span>
                  </div>
                  <div className="border-t border-border pt-3">
                    <div className="flex justify-between text-sm mb-2">
                      <span className="text-muted">Overall Score</span>
                      <span className="font-bold text-amber-600">62/100</span>
                    </div>
                    <Progress value={62} />
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-xl border border-border shadow-sm p-6">
                <h4 className="font-semibold text-foreground mb-3">Missing Clauses</h4>
                <ul className="space-y-2 text-sm">
                  {["Force Majeure", "Non-Compete", "Audit Rights"].map((clause) => (
                    <li key={clause} className="flex items-center gap-2 text-muted">
                      <div className="h-1.5 w-1.5 rounded-full bg-amber-400" />
                      {clause}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </TabsContent>

        {/* Draft Assistant Tab */}
        <TabsContent value="draft-assistant">
          <div className="max-w-3xl">
            <div className="bg-white rounded-xl border border-border shadow-sm p-6">
              <h3 className="text-lg font-semibold text-foreground mb-2">Draft Assistant</h3>
              <p className="text-sm text-muted mb-6">
                Generate legal documents with inline source references from your matter documents.
              </p>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-foreground mb-1.5">Document Type</label>
                  <Select value={draftType} onValueChange={setDraftType}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select document type" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="brief">Legal Brief</SelectItem>
                      <SelectItem value="memo">Legal Memorandum</SelectItem>
                      <SelectItem value="motion">Motion</SelectItem>
                      <SelectItem value="response">Response Letter</SelectItem>
                      <SelectItem value="summary">Executive Summary</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-foreground mb-1.5">Instructions</label>
                  <textarea
                    placeholder="Describe what you need drafted. Be specific about the audience, key points to cover, and any constraints..."
                    className="flex min-h-[120px] w-full rounded-md border border-input bg-white px-3 py-2 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-accent/20 focus:border-accent resize-none"
                  />
                </div>

                <Button disabled={!draftType}>
                  <PenLine className="h-4 w-4" />
                  Generate Draft
                </Button>
              </div>
            </div>
          </div>
        </TabsContent>

        {/* Audit Log Tab */}
        <TabsContent value="audit-log">
          <div className="bg-white rounded-xl border border-border shadow-sm">
            <div className="p-4 border-b border-border flex items-center justify-between">
              <h3 className="font-semibold text-foreground">Activity Log</h3>
              <Button variant="outline" size="sm">
                <Download className="h-4 w-4" />
                Export Log
              </Button>
            </div>
            <DataTable columns={auditColumns} data={mockAuditLog} />
          </div>
        </TabsContent>
      </Tabs>
    </AppLayout>
  )
}
