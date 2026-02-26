"use client"

import React, { useState } from "react"
import { motion } from "framer-motion"
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
    <div className="min-h-screen bg-background">
      <Sidebar collapsed={collapsed} onToggle={() => setCollapsed(!collapsed)} />
      <div
        className={cn(
          "transition-all duration-300 ease-[cubic-bezier(0.16,1,0.3,1)]",
          collapsed ? "ml-[68px]" : "ml-[260px]"
        )}
      >
        <Topbar title={title} />
        <motion.main
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
          className="p-8"
        >
          {children}
        </motion.main>
      </div>
    </div>
  )
}
