"use client"

import React, { useState } from "react"
import {
  Users,
  Plus,
  Mail,
  Shield,
  MoreHorizontal,
  UserCheck,
  UserX,
  Clock,
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

interface TeamMember {
  id: string
  name: string
  email: string
  role: string
  matters: number
  queries: number
  lastActive: string
  status: string
  [key: string]: unknown
}

const mockTeam: TeamMember[] = [
  { id: "t1", name: "John Smith", email: "john.smith@firm.com", role: "Partner", matters: 8, queries: 156, lastActive: new Date(Date.now() - 600000).toISOString(), status: "active" },
  { id: "t2", name: "Sarah Chen", email: "sarah.chen@firm.com", role: "Associate", matters: 5, queries: 89, lastActive: new Date(Date.now() - 3600000).toISOString(), status: "active" },
  { id: "t3", name: "Lisa Park", email: "lisa.park@firm.com", role: "Paralegal", matters: 6, queries: 67, lastActive: new Date(Date.now() - 7200000).toISOString(), status: "active" },
  { id: "t4", name: "Mike Torres", email: "mike.torres@firm.com", role: "Associate", matters: 4, queries: 45, lastActive: new Date(Date.now() - 86400000).toISOString(), status: "active" },
  { id: "t5", name: "Emily Johnson", email: "emily.j@firm.com", role: "Paralegal", matters: 0, queries: 0, lastActive: "", status: "invited" },
  { id: "t6", name: "Robert Williams", email: "r.williams@firm.com", role: "Partner", matters: 12, queries: 234, lastActive: new Date(Date.now() - 2592000000).toISOString(), status: "deactivated" },
]

export default function TeamPage() {
  const [showInviteDialog, setShowInviteDialog] = useState(false)
  const [inviteEmail, setInviteEmail] = useState("")
  const [inviteRole, setInviteRole] = useState("")

  const columns = [
    {
      key: "name",
      header: "Member",
      render: (item: TeamMember) => (
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-full bg-foreground text-white font-medium text-sm">
            {item.name.split(" ").map((n) => n[0]).join("")}
          </div>
          <div>
            <p className="font-medium text-foreground">{item.name}</p>
            <p className="text-xs text-muted">{item.email}</p>
          </div>
        </div>
      ),
    },
    {
      key: "role",
      header: "Role",
      render: (item: TeamMember) => (
        <div className="flex items-center gap-1.5">
          <Shield className="h-3.5 w-3.5 text-muted" />
          <span className="text-sm">{item.role}</span>
        </div>
      ),
    },
    {
      key: "matters",
      header: "Matters",
      render: (item: TeamMember) => <span className="text-sm font-medium">{item.matters}</span>,
    },
    {
      key: "queries",
      header: "Queries",
      render: (item: TeamMember) => <span className="text-sm font-mono">{item.queries}</span>,
    },
    {
      key: "status",
      header: "Status",
      render: (item: TeamMember) => {
        if (item.status === "active") return <Badge variant="active"><UserCheck className="h-3 w-3 mr-1" />Active</Badge>
        if (item.status === "invited") return <Badge variant="review"><Clock className="h-3 w-3 mr-1" />Invited</Badge>
        return <Badge variant="closed"><UserX className="h-3 w-3 mr-1" />Deactivated</Badge>
      },
    },
    {
      key: "lastActive",
      header: "Last Active",
      render: (item: TeamMember) => (
        <span className="text-sm text-muted">
          {item.lastActive ? formatRelativeTime(item.lastActive) : "Never"}
        </span>
      ),
    },
    {
      key: "actions",
      header: "",
      className: "w-12",
      render: () => (
        <Button variant="ghost" size="icon" className="h-8 w-8">
          <MoreHorizontal className="h-4 w-4 text-muted" />
        </Button>
      ),
    },
  ]

  const activeCount = mockTeam.filter((m) => m.status === "active").length
  const invitedCount = mockTeam.filter((m) => m.status === "invited").length

  return (
    <AppLayout title="Team">
      <PageHeader
        title="Team Management"
        description={`${activeCount} active members, ${invitedCount} pending invitations`}
        actions={
          <Button onClick={() => setShowInviteDialog(true)}>
            <Plus className="h-4 w-4" />
            Invite Member
          </Button>
        }
      />

      {/* Role Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        {[
          { role: "Partners", count: mockTeam.filter((m) => m.role === "Partner").length, desc: "Full access to all matters and settings" },
          { role: "Associates", count: mockTeam.filter((m) => m.role === "Associate").length, desc: "Access to assigned matters" },
          { role: "Paralegals", count: mockTeam.filter((m) => m.role === "Paralegal").length, desc: "Document upload and basic queries" },
        ].map((r) => (
          <div key={r.role} className="bg-white rounded-sm border border-border p-6 shadow-sm">
            <div className="flex items-center justify-between">
              <p className="text-sm font-medium text-muted">{r.role}</p>
              <span className="text-2xl font-bold text-foreground">{r.count}</span>
            </div>
            <p className="text-xs text-muted mt-1">{r.desc}</p>
          </div>
        ))}
      </div>

      {/* Team Table */}
      <div className="bg-white rounded-sm border border-border shadow-sm">
        <DataTable columns={columns} data={mockTeam} />
      </div>

      {/* Invite Dialog */}
      <Dialog open={showInviteDialog} onOpenChange={setShowInviteDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Invite Team Member</DialogTitle>
            <DialogDescription>
              Send an invitation to join your LexIntel workspace.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 mt-4">
            <div>
              <label className="block text-sm font-medium text-foreground mb-1.5">Email Address</label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted" />
                <Input
                  type="email"
                  placeholder="colleague@firm.com"
                  value={inviteEmail}
                  onChange={(e) => setInviteEmail(e.target.value)}
                  className="pl-9"
                />
              </div>
            </div>
            <div>
              <label className="block text-sm font-medium text-foreground mb-1.5">Role</label>
              <Select value={inviteRole} onValueChange={setInviteRole}>
                <SelectTrigger>
                  <SelectValue placeholder="Select role" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="partner">Partner</SelectItem>
                  <SelectItem value="associate">Associate</SelectItem>
                  <SelectItem value="paralegal">Paralegal</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowInviteDialog(false)}>
              Cancel
            </Button>
            <Button onClick={() => setShowInviteDialog(false)}>
              <Mail className="h-4 w-4" />
              Send Invitation
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </AppLayout>
  )
}
