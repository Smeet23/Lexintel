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
  { action: "Query answered", matter: "Acme vs Global Corp", user: "John Smith", time: "10 min ago" },
  { action: "Document uploaded", matter: "Smith Estate Planning", user: "Sarah Chen", time: "1 hour ago" },
  { action: "Contract reviewed", matter: "TechStart IP Review", user: "John Smith", time: "3 hours ago" },
  { action: "New matter created", matter: "Metro Construction", user: "Lisa Park", time: "5 hours ago" },
  { action: "Draft exported", matter: "Phoenix Merger", user: "Mike Torres", time: "Yesterday" },
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
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent/10">
            <Briefcase className="h-4 w-4 text-accent" />
          </div>
          <div>
            <p className="font-medium text-foreground">{item.title}</p>
            <p className="text-xs text-muted">{item.documentsCount} documents</p>
          </div>
        </div>
      ),
    },
    { key: "jurisdiction", header: "Jurisdiction" },
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
        <span className="text-muted">{formatRelativeTime(item.lastActivity)}</span>
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
            <Briefcase className="h-4 w-4" />
            New Matter
          </Button>
        }
      />

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        {stats.map((stat) => (
          <StatsCard key={stat.title} {...stat} />
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Recent Matters */}
        <div className="lg:col-span-2 bg-white rounded-xl border border-border shadow-sm">
          <div className="flex items-center justify-between p-6 pb-0">
            <h3 className="text-lg font-semibold text-foreground">Recent Matters</h3>
            <Button
              variant="ghost"
              size="sm"
              className="text-accent"
              onClick={() => router.push("/matters")}
            >
              View All <ArrowRight className="h-4 w-4 ml-1" />
            </Button>
          </div>
          <div className="p-6">
            <DataTable
              columns={matterColumns}
              data={recentMatters}
              onRowClick={(item) => router.push(`/matters/${item.id}`)}
            />
          </div>
        </div>

        {/* Recent Activity */}
        <div className="bg-white rounded-xl border border-border shadow-sm">
          <div className="p-6 pb-4">
            <h3 className="text-lg font-semibold text-foreground">Recent Activity</h3>
          </div>
          <div className="px-6 pb-6 space-y-4">
            {recentActivity.map((activity, idx) => (
              <div key={idx} className="flex items-start gap-3">
                <div className="mt-0.5">
                  <div className="h-8 w-8 rounded-full bg-surface flex items-center justify-center">
                    {activity.action.includes("Query") && <MessageSquare className="h-4 w-4 text-accent" />}
                    {activity.action.includes("Document") && <FileText className="h-4 w-4 text-emerald-600" />}
                    {activity.action.includes("Contract") && <AlertCircle className="h-4 w-4 text-amber-600" />}
                    {activity.action.includes("matter") && <Briefcase className="h-4 w-4 text-violet-600" />}
                    {activity.action.includes("Draft") && <FileText className="h-4 w-4 text-blue-600" />}
                  </div>
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-foreground">
                    <span className="font-medium">{activity.action}</span>
                  </p>
                  <p className="text-xs text-muted truncate">
                    {activity.matter} &middot; {activity.user}
                  </p>
                </div>
                <span className="text-xs text-muted whitespace-nowrap flex items-center gap-1">
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
