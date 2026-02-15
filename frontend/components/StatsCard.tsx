import React from "react"
import { cn } from "@/lib/utils"
import { type LucideIcon } from "lucide-react"

interface StatsCardProps {
  title: string
  value: string | number
  icon: LucideIcon
  trend?: { value: string; positive: boolean }
  className?: string
}

export default function StatsCard({ title, value, icon: Icon, trend, className }: StatsCardProps) {
  return (
    <div className={cn(
      "bg-white rounded-xl border border-border p-6 transition-all duration-300 hover:shadow-elevated hover:border-border-strong group",
      className
    )}>
      <div className="flex items-start justify-between">
        <div>
          <p className="text-[12px] font-medium uppercase tracking-[0.06em] text-muted">{title}</p>
          <p className="font-display text-[28px] text-foreground mt-2 tracking-tight">{value}</p>
          {trend && (
            <p className={cn(
              "text-[11px] font-medium mt-2",
              trend.positive ? "text-emerald-600" : "text-red-600"
            )}>
              {trend.positive ? "+" : ""}{trend.value} from last month
            </p>
          )}
        </div>
        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-surface group-hover:bg-surface-hover transition-colors">
          <Icon className="h-[18px] w-[18px] text-muted" />
        </div>
      </div>
    </div>
  )
}
