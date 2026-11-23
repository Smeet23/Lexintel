import React from "react"

interface PageHeaderProps {
  title: string
  description?: string
  actions?: React.ReactNode
}

export default function PageHeader({ title, description, actions }: PageHeaderProps) {
  return (
    <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4 mb-8 lg:mb-10">
      <div>
        <h2 className="font-display text-[24px] sm:text-[28px] text-foreground tracking-tight">{title}</h2>
        {description && (
          <p className="mt-1.5 text-[13px] sm:text-[14px] text-muted">{description}</p>
        )}
      </div>
      {actions && <div className="flex items-center gap-2 shrink-0">{actions}</div>}
    </div>
  )
}
