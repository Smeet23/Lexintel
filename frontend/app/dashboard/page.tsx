"use client"

import React from "react"
import { useRouter } from "next/navigation"
import {
  Briefcase,
  Zap,
  AlertCircle,
  MessageSquare,
  ArrowRight,
  FileText,
  Clock,
} from "lucide-react"
import AppLayout from "@/layouts/AppLayout"
import StatsCard from "@/components/StatsCard"
import PageHeader from "@/components/PageHeader"
import DataTable from "@/components/DataTable"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { formatRelativeTime } from "@/lib/utils"

const stats = [
  { title: "Active Matters", value: "18", icon: Briefcase, trend: { value: "3", positive: true } },
  { title: "Token Usage", value: "42,390", icon: Zap, trend: { value: "12%", positive: true } },
  { title: "Pending Reviews", value: "7", icon: AlertCircle, trend: { value: "2", positive: false } },
  { title: "Queries This Week", value: "156", icon: MessageSquare, trend: { value: "18%", positive: true } },
]

const recentMatters = [
  { id: "1", title: "Acme vs Global Corp", jurisdiction: "US - Federal", status: "active" as const, lastActivity: new Date(Date.now() - 7200000).toISOString(), documentsCount: 12 },
  { id: "2", title: "Smith Estate Planning", jurisdiction: "US - California", status: "active" as const, lastActivity: new Date(Date.now() - 18000000).toISOString(), documentsCount: 8 },
  { id: "3", title: "TechStart IP Review", jurisdiction: "US - Delaware", status: "review" as const, lastActivity: new Date(Date.now() - 86400000).toISOString(), documentsCount: 23 },
  { id: "4", title: "Metro Construction Dispute", jurisdiction: "UK", status: "active" as const, lastActivity: new Date(Date.now() - 172800000).toISOString(), documentsCount: 15 },
  { id: "5", title: "Phoenix Merger Analysis", jurisdiction: "EU", status: "closed" as const, lastActivity: new Date(Date.now() - 604800000).toISOString(), documentsCount: 34 },
]

const recentActivity = [
  { action: "Query answered", matter: "Acme vs Global Corp", user: "J. Smith", time: "10 min ago", icon: MessageSquare },
  { action: "Document uploaded", matter: "Smith Estate Planning", user: "S. Chen", time: "1 hour ago", icon: FileText },
  { action: "Contract reviewed", matter: "TechStart IP Review", user: "J. Smith", time: "3 hours ago", icon: AlertCircle },
  { action: "New matter created", matter: "Metro Construction", user: "L. Park", time: "5 hours ago", icon: Briefcase },
  { action: "Draft exported", matter: "Phoenix Merger", user: "M. Torres", time: "Yesterday", icon: FileText },
]

const statusMap: Record<string, "active" | "review" | "closed"> = {
  active: "active",
  review: "review",
  closed: "closed",
}

export default function DashboardPage() {
  const router = useRouter()

  const matterColumns = [
    {
      key: "title",
      header: "Matter",
      render: (item: typeof recentMatters[0]) => (
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-sm bg-surface">
            <Briefcase className="h-3.5 w-3.5 text-muted" />
          </div>
          <div>
            <p className="text-[13px] font-medium text-foreground">{item.title}</p>
            <p className="text-[11px] text-muted">{item.documentsCount} documents</p>
          </div>
        </div>
      ),
    },
    {
      key: "jurisdiction",
      header: "Jurisdiction",
      render: (item: typeof recentMatters[0]) => <span className="text-[13px] text-muted">{item.jurisdiction}</span>,
    },
    {
      key: "status",
      header: "Status",
      render: (item: typeof recentMatters[0]) => (
        <Badge variant={statusMap[item.status]}>
          {item.status.charAt(0).toUpperCase() + item.status.slice(1)}
        </Badge>
      ),
    },
    {
      key: "lastActivity",
      header: "Last Activity",
      render: (item: typeof recentMatters[0]) => (
        <span className="text-[13px] text-muted">{formatRelativeTime(item.lastActivity)}</span>
      ),
    },
  ]

  return (
    <AppLayout title="Dashboard">
      <PageHeader
        title="Dashboard"
        description="Overview of your legal workspace"
        actions={
          <Button onClick={() => router.push("/matters")}>
            <Briefcase className="h-3.5 w-3.5" />
            New Matter
          </Button>
        }
      />

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5 mb-10">
        {stats.map((stat) => (
          <StatsCard key={stat.title} {...stat} />
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Recent Matters */}
        <div className="lg:col-span-2 bg-white rounded-sm border border-border">
          <div className="flex items-center justify-between px-6 py-5 border-b border-border">
            <h3 className="font-display text-[16px] text-foreground">Recent Matters</h3>
            <Button
              variant="ghost"
              size="sm"
              className="text-muted hover:text-foreground"
              onClick={() => router.push("/matters")}
            >
              View All <ArrowRight className="h-3.5 w-3.5 ml-1" />
            </Button>
          </div>
          <DataTable
            columns={matterColumns}
            data={recentMatters}
            onRowClick={(item) => router.push(`/matters/${item.id}`)}
          />
        </div>

        {/* Activity Feed */}
        <div className="bg-white rounded-sm border border-border">
          <div className="px-6 py-5 border-b border-border">
            <h3 className="font-display text-[16px] text-foreground">Activity</h3>
          </div>
          <div className="px-6 py-5 space-y-5">
            {recentActivity.map((activity, idx) => (
              <div key={idx} className="flex items-start gap-3.5">
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-sm bg-surface mt-0.5">
                  <activity.icon className="h-3.5 w-3.5 text-muted" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-[13px] text-foreground leading-relaxed">
                    <span className="font-medium">{activity.action}</span>
                  </p>
                  <p className="text-[11px] text-muted truncate mt-0.5">
                    {activity.matter} &middot; {activity.user}
                  </p>
                </div>
                <span className="text-[11px] text-muted-foreground whitespace-nowrap flex items-center gap-1.5 mt-0.5">
                  <Clock className="h-3 w-3" />
                  {activity.time}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </AppLayout>
  )
}
