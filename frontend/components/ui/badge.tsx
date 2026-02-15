import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "@/lib/utils"

const badgeVariants = cva(
  "inline-flex items-center rounded-md px-2.5 py-0.5 text-[11px] font-medium tracking-wide transition-colors",
  {
    variants: {
      variant: {
        default: "bg-primary/5 text-primary border border-primary/10",
        active: "bg-emerald-50 text-emerald-700 border border-emerald-200/60",
        review: "bg-amber-50 text-amber-700 border border-amber-200/60",
        closed: "bg-surface text-muted-foreground border border-border",
        error: "bg-red-50 text-red-700 border border-red-200/60",
        processing: "bg-blue-50 text-blue-700 border border-blue-200/60",
        indexed: "bg-emerald-50 text-emerald-700 border border-emerald-200/60",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
)

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return <div className={cn(badgeVariants({ variant }), className)} {...props} />
}

export { Badge, badgeVariants }
