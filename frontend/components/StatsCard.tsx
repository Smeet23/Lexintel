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
    <div className={cn("bg-white rounded-xl border border-border p-6 shadow-sm hover:shadow-md transition-shadow", className)}>
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm font-medium text-muted">{title}</p>
          <p className="text-3xl font-bold text-foreground mt-2">{value}</p>
          {trend && (
            <p
              className={cn(
                "text-xs font-medium mt-2",
                trend.positive ? "text-emerald-600" : "text-red-500"
              )}
            >
              {trend.positive ? "+" : ""}{trend.value} from last month
            </p>
          )}
        </div>
        <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-accent/10">
          <Icon className="h-5 w-5 text-accent" />
        </div>
      </div>
    </div>
  )
}
