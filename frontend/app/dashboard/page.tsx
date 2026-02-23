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
  Loader2,
} from "lucide-react"
import AppLayout from "@/layouts/AppLayout"
import StatsCard from "@/components/StatsCard"
import PageHeader from "@/components/PageHeader"
import DataTable from "@/components/DataTable"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { formatRelativeTime } from "@/lib/utils"
import { caseToMatter } from "@/lib/types"
import { useCases } from "@/hooks/use-cases"

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
  const { data: cases, isLoading } = useCases()

  const matters = (cases || []).map(caseToMatter)
  const recentMatters = [...matters]
    .sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime())
    .slice(0, 5)

  const activeCount = matters.filter((m) => m.status === "active").length
  const processingCount = matters.filter((m) => m.status === "review").length

  const stats = [
    { title: "Active Matters", value: String(activeCount), icon: Briefcase, trend: { value: String(matters.length), positive: true } },
    { title: "Token Usage", value: "—", icon: Zap },
    { title: "Processing", value: String(processingCount), icon: AlertCircle },
    { title: "Total Matters", value: String(matters.length), icon: MessageSquare },
  ]

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
    {
      key: "status",
      header: "Status",
      render: (item: typeof recentMatters[0]) => (
        <Badge variant={statusMap[item.status] || "active"}>
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
            {isLoading ? (
              <div className="flex items-center justify-center py-10">
                <Loader2 className="h-5 w-5 animate-spin text-accent" />
                <span className="ml-2 text-sm text-muted">Loading...</span>
              </div>
            ) : (
              <DataTable
                columns={matterColumns}
                data={recentMatters}
                onRowClick={(item) => router.push(`/matters/${item.id}`)}
                emptyMessage="No matters yet. Create your first matter to get started."
              />
            )}
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
