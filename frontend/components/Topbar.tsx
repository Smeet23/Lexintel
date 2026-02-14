"use client"

import React from "react"
import { Bell, Search, CircleUser } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"

interface TopbarProps {
  title: string
}

export default function Topbar({ title }: TopbarProps) {
  return (
    <header className="sticky top-0 z-30 flex h-16 items-center justify-between border-b border-border bg-white px-6">
      <h1 className="text-lg font-semibold text-foreground">{title}</h1>

      <div className="flex items-center gap-3">
        {/* Search */}
        <div className="relative hidden md:block">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted" />
          <Input
            placeholder="Search matters, documents..."
            className="w-64 pl-9 h-9 bg-surface border-border"
          />
        </div>

        {/* Notifications */}
        <Button variant="ghost" size="icon" className="relative">
          <Bell className="h-5 w-5 text-muted" />
          <span className="absolute right-1.5 top-1.5 h-2 w-2 rounded-full bg-accent" />
        </Button>

        {/* User */}
        <div className="flex items-center gap-2 rounded-lg px-2 py-1.5 hover:bg-surface-hover transition-colors cursor-pointer">
          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-accent/10">
            <CircleUser className="h-5 w-5 text-accent" />
          </div>
          <div className="hidden md:block">
            <p className="text-sm font-medium text-foreground leading-none">John Smith</p>
            <p className="text-xs text-muted mt-0.5">Partner</p>
          </div>
        </div>
      </div>
    </header>
  )
}
