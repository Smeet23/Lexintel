import type { LucideIcon } from "lucide-react"
import { Clock } from "lucide-react"

interface ComingSoonProps {
  icon: LucideIcon
  description: string
}

export default function ComingSoon({ icon: Icon, description }: ComingSoonProps) {
  return (
    <div className="flex justify-center py-12 sm:py-16">
      <div className="w-full max-w-xl bg-white rounded-xl border border-border shadow-elevated p-8 sm:p-10 text-center">
        <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-surface mx-auto mb-4">
          <Icon className="h-5 w-5 text-muted" />
        </div>
        <h2 className="font-display text-[18px] text-foreground">Coming soon</h2>
        <p className="text-sm text-muted mt-2">{description}</p>
      </div>
    </div>
  )
}
