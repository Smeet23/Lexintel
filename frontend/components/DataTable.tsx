import React from "react"
import { cn } from "@/lib/utils"

/* eslint-disable @typescript-eslint/no-explicit-any */

interface Column {
  key: string
  header: string
  className?: string
  render?: (item: any) => React.ReactNode
}

interface DataTableProps {
  columns: Column[]
  data: any[]
  onRowClick?: (item: any) => void
  emptyMessage?: string
}

export default function DataTable({
  columns,
  data,
  onRowClick,
  emptyMessage = "No data found",
}: DataTableProps) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left">
        <thead>
          <tr className="border-b border-border">
            {columns.map((col) => (
              <th
                key={col.key}
                className={cn(
                  "py-3 px-4 text-xs font-semibold uppercase tracking-wider text-muted",
                  col.className
                )}
              >
                {col.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.length === 0 ? (
            <tr>
              <td
                colSpan={columns.length}
                className="py-12 text-center text-sm text-muted"
              >
                {emptyMessage}
              </td>
            </tr>
          ) : (
            data.map((item, idx) => (
              <tr
                key={idx}
                onClick={() => onRowClick?.(item)}
                className={cn(
                  "border-b border-border last:border-0 transition-colors",
                  onRowClick && "cursor-pointer hover:bg-surface-hover"
                )}
              >
                {columns.map((col) => (
                  <td key={col.key} className={cn("py-3.5 px-4 text-sm", col.className)}>
                    {col.render
                      ? col.render(item)
                      : (item[col.key] as React.ReactNode)}
                  </td>
                ))}
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  )
}
