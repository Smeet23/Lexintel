"use client"

import React, { useState } from "react"
import Sidebar from "@/components/Sidebar"
import Topbar from "@/components/Topbar"
import { cn } from "@/lib/utils"

interface AppLayoutProps {
  children: React.ReactNode
  title: string
}

export default function AppLayout({ children, title }: AppLayoutProps) {
  const [collapsed, setCollapsed] = useState(false)

  return (
    <div className="min-h-screen bg-surface">
      <Sidebar collapsed={collapsed} onToggle={() => setCollapsed(!collapsed)} />
      <div
        className={cn(
          "transition-all duration-300",
          collapsed ? "ml-[72px]" : "ml-64"
        )}
      >
        <Topbar title={title} />
        <main className="p-6">{children}</main>
      </div>
    </div>
  )
}
