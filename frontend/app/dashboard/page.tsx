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
import { useMatters } from "@/hooks/use-matters"
import type { MatterResponse } from "@/lib/api-services"

const recentActivity = [
  { action: "Query answered", matter: "Acme vs Global Corp", user: "J. Smith", time: "10 min ago", icon: MessageSquare },
  { action: "Document uploaded", matter: "Smith Estate Planning", user: "S. Chen", time: "1 hour ago", icon: FileText },
  { action: "Contract reviewed", matter: "TechStart IP Review", user: "J. Smith", time: "3 hours ago", icon: AlertCircle },
  { action: "New matter created", matter: "Metro Construction", user: "L. Park", time: "5 hours ago", icon: Briefcase },
  { action: "Draft exported", matter: "Phoenix Merger", user: "M. Torres", time: "Yesterday", icon: FileText },
]

function mapStatus(status: string): "active" | "review" | "closed" {
  if (status === "ready") return "active"
  if (status === "processing") return "review"
  return "closed"
}

export default function DashboardPage() {
  const router = useRouter()
  const { data: matters, isLoading, error } = useMatters()

  const recentMatters = (matters || []).slice(0, 5)

  // Compute stats from real data
  const activeCount = (matters || []).filter(m => m.status === "ready").length
  const processingCount = (matters || []).filter(m => m.status === "processing").length
  const totalMatters = (matters || []).length

  const stats = [
    { title: "Active Matters", value: String(activeCount), icon: Briefcase, trend: { value: String(totalMatters) + " total", positive: true } },
    { title: "Processing", value: String(processingCount), icon: Zap, trend: processingCount > 0 ? { value: "in progress", positive: true } : undefined },
    { title: "Total Matters", value: String(totalMatters), icon: AlertCircle },
    { title: "Error", value: String((matters || []).filter(m => m.status === "error").length), icon: MessageSquare },
  ]

  const matterColumns = [
    {
      key: "title",
      header: "Matter",
      render: (item: MatterResponse) => (
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-sm bg-surface">
            <Briefcase className="h-3.5 w-3.5 text-muted" />
          </div>
          <div>
            <p className="text-[13px] font-medium text-foreground">{item.name}</p>
            <p className="text-[11px] text-muted">{item.file_type.toUpperCase()}</p>
          </div>
        </div>
      ),
    },
    {
      key: "status",
      header: "Status",
      render: (item: MatterResponse) => (
        <Badge variant={mapStatus(item.status)}>
          {item.status === "ready" ? "Ready" : item.status === "processing" ? "Processing" : item.status === "error" ? "Error" : item.status}
        </Badge>
      ),
    },
    {
      key: "lastActivity",
      header: "Last Activity",
      render: (item: MatterResponse) => (
        <span className="text-[13px] text-muted">{item.updated_at ? formatRelativeTime(item.updated_at) : formatRelativeTime(item.created_at)}</span>
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
          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-5 w-5 animate-spin text-muted" />
              <span className="ml-2 text-sm text-muted">Loading matters...</span>
            </div>
          ) : error ? (
            <div className="px-6 py-12 text-center">
              <p className="text-sm text-muted">Failed to load matters. Is the backend running?</p>
            </div>
          ) : recentMatters.length === 0 ? (
            <div className="px-6 py-12 text-center">
              <Briefcase className="h-8 w-8 text-muted mx-auto mb-3" />
              <p className="text-sm font-medium text-foreground">No matters yet</p>
              <p className="text-xs text-muted mt-1">Create your first matter to get started</p>
            </div>
          ) : (
            <DataTable
              columns={matterColumns}
              data={recentMatters}
              onRowClick={(item) => router.push(`/matters/${item.id}`)}
            />
          )}
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
