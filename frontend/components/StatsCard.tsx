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
      "bg-white rounded-xl border border-border p-4 sm:p-6 transition-all duration-300 hover:shadow-elevated hover:border-border-strong group",
      className
    )}>
      <div className="flex items-start justify-between">
        <div className="min-w-0">
          <p className="text-[11px] sm:text-[12px] font-medium uppercase tracking-[0.06em] text-muted truncate">{title}</p>
          <p className="font-display text-[22px] sm:text-[28px] text-foreground mt-1.5 sm:mt-2 tracking-tight">{value}</p>
          <p className={cn(
            "text-[10px] sm:text-[11px] font-medium mt-1.5 sm:mt-2",
            trend ? (trend.positive ? "text-emerald-600" : "text-red-600") : "text-transparent select-none"
          )}>
            {trend ? `${trend.positive ? "+" : ""}${trend.value} from last month` : "\u00A0"}
          </p>
        </div>
        <div className="flex h-8 w-8 sm:h-10 sm:w-10 items-center justify-center rounded-xl bg-surface group-hover:bg-surface-hover transition-colors shrink-0">
          <Icon className="h-4 w-4 sm:h-[18px] sm:w-[18px] text-muted" />
        </div>
      </div>
    </div>
  )
}
