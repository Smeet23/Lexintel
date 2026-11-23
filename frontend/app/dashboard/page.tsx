"use client"

import React from "react"
import { useRouter } from "next/navigation"
import { motion } from "framer-motion"
import {
  Briefcase,
  Zap,
  AlertCircle,
  FolderOpen,
  ArrowRight,
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
import ErrorState from "@/components/ErrorState"

function mapStatus(status: string): "active" | "review" | "closed" {
  if (status === "ready") return "active"
  if (status === "processing") return "review"
  return "closed"
}

const stagger = {
  hidden: {},
  show: { transition: { staggerChildren: 0.06 } },
}

const fadeUp = {
  hidden: { opacity: 0, y: 12 },
  show: { opacity: 1, y: 0, transition: { duration: 0.4, ease: [0.16, 1, 0.3, 1] as const } },
}

export default function DashboardPage() {
  const router = useRouter()
  const { data: matters, isLoading, isError, refetch } = useMatters()

  const recentMatters = (matters || []).slice(0, 5)

  const activeCount = matters?.filter(m => m.status === "ready").length
  const processingCount = matters?.filter(m => m.status === "processing").length
  const totalMatters = matters?.length

  const stats = [
    { title: "Active Matters", value: isError ? "—" : String(activeCount ?? 0), icon: Briefcase },
    { title: "Processing", value: isError ? "—" : String(processingCount ?? 0), icon: Zap },
    { title: "Total Matters", value: isError ? "—" : String(totalMatters ?? 0), icon: FolderOpen },
    { title: "Error", value: isError ? "—" : String(matters?.filter(m => m.status === "error").length ?? 0), icon: AlertCircle },
  ]

  const matterColumns = [
    {
      key: "title",
      header: "Matter",
      render: (item: MatterResponse) => (
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-surface">
            <Briefcase className="h-4 w-4 text-muted" />
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
          {item.status === "ready" ? "Ready" : item.status === "processing" ? "Processing" : item.status === "cancelled" ? "Cancelled" : item.status === "error" ? "Error" : item.status}
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
      <motion.div
        variants={stagger}
        initial="hidden"
        animate="show"
        className="grid grid-cols-2 md:grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-5 mb-8 lg:mb-10"
      >
        {stats.map((stat) => (
          <motion.div key={stat.title} variants={fadeUp} className="h-full">
            <StatsCard {...stat} />
          </motion.div>
        ))}
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 lg:gap-6">
        {/* Recent Matters */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2, ease: [0.16, 1, 0.3, 1] }}
          className="lg:col-span-2 bg-white rounded-xl border border-border shadow-elevated"
        >
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
          ) : isError ? (
            <ErrorState onRetry={() => refetch()} />
          ) : recentMatters.length === 0 ? (
            <div className="px-6 py-12 text-center">
              <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-surface mx-auto mb-3">
                <Briefcase className="h-5 w-5 text-muted" />
              </div>
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
        </motion.div>

        {/* Activity Feed */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3, ease: [0.16, 1, 0.3, 1] }}
          className="bg-white rounded-xl border border-border shadow-elevated"
        >
          <div className="px-6 py-5 border-b border-border">
            <h3 className="font-display text-[16px] text-foreground">Activity</h3>
          </div>
          <div className="px-6 py-8 text-center">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-surface mx-auto mb-3">
              <Briefcase className="h-4 w-4 text-muted" />
            </div>
            <p className="text-sm font-medium text-foreground">No recent activity</p>
            <p className="text-xs text-muted mt-1">Activity from your matters will appear here.</p>
          </div>
        </motion.div>
      </div>
    </AppLayout>
  )
}
