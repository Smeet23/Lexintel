"use client"

import React from "react"
import { Bell, Search, ChevronDown } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"

interface TopbarProps {
  title: string
}

export default function Topbar({ title }: TopbarProps) {
  return (
    <header className="sticky top-0 z-30 flex h-[60px] items-center justify-between border-b border-border/60 glass px-8">
      <h1 className="text-[14px] font-medium text-foreground tracking-tight">{title}</h1>

      <div className="flex items-center gap-2">
        {/* Search */}
        <div className="relative hidden md:block">
          <Search className="absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted" />
          <Input
            placeholder="Search..."
            className="w-56 pl-8 h-8 text-[13px] bg-white/60 border-transparent focus:border-border focus:bg-white rounded-lg"
          />
        </div>

        {/* Notifications */}
        <Button variant="ghost" size="icon" className="relative h-8 w-8 rounded-lg hover:bg-white/80">
          <Bell className="h-4 w-4 text-muted" />
          <span className="absolute right-1.5 top-1.5 h-1.5 w-1.5 rounded-full bg-foreground ring-2 ring-white" />
        </Button>

        {/* Divider */}
        <div className="h-5 w-px bg-border mx-1" />

        {/* User */}
        <button className="flex items-center gap-2 rounded-lg px-2 py-1.5 hover:bg-white/80 transition-all duration-200 cursor-pointer">
          <div className="flex h-7 w-7 items-center justify-center rounded-full bg-foreground text-[11px] font-semibold text-white">
            JS
          </div>
          <div className="hidden md:block text-left">
            <p className="text-[13px] font-medium text-foreground leading-none">John Smith</p>
            <p className="text-[11px] text-muted mt-0.5">Partner</p>
          </div>
          <ChevronDown className="h-3 w-3 text-muted hidden md:block" />
        </button>
      </div>
    </header>
  )
}
