"use client"

import React from "react"
import { Menu } from "lucide-react"
import { Button } from "@/components/ui/button"

interface TopbarProps {
  title: string
  onMobileMenuToggle?: () => void
}

export default function Topbar({ title, onMobileMenuToggle }: TopbarProps) {
  return (
    <header className="sticky top-0 z-30 flex h-[60px] items-center justify-between border-b border-border/60 glass px-4 sm:px-6 lg:px-8">
      <div className="flex items-center gap-3">
        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8 rounded-lg lg:hidden"
          onClick={onMobileMenuToggle}
        >
          <Menu className="h-4 w-4 text-foreground" />
        </Button>
        <h1 className="text-[14px] font-medium text-foreground tracking-tight">{title}</h1>
      </div>
    </header>
  )
}
