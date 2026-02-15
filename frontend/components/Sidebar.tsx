"use client"

import React from "react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import {
  LayoutDashboard,
  Briefcase,
  BookOpen,
  Users,
  CreditCard,
  Settings,
  Scale,
  LogOut,
  ChevronLeft,
  ChevronRight,
} from "lucide-react"
import { cn } from "@/lib/utils"

const navigation = [
  { name: "Dashboard", href: "/dashboard", icon: LayoutDashboard },
  { name: "Matters", href: "/matters", icon: Briefcase },
  { name: "Precedents", href: "/precedents", icon: BookOpen },
  { name: "Team", href: "/team", icon: Users },
  { name: "Billing", href: "/billing", icon: CreditCard },
  { name: "Settings", href: "/settings", icon: Settings },
]

interface SidebarProps {
  collapsed?: boolean
  onToggle?: () => void
}

export default function Sidebar({ collapsed = false, onToggle }: SidebarProps) {
  const pathname = usePathname()

  return (
    <aside
      className={cn(
        "fixed left-0 top-0 z-40 flex h-screen flex-col bg-white border-r border-border transition-all duration-300",
        collapsed ? "w-[68px]" : "w-[260px]"
      )}
    >
      {/* Logo */}
      <div className="flex h-[60px] items-center gap-3 px-5 border-b border-border">
        <div className="flex h-8 w-8 shrink-0 items-center justify-center">
          <Scale className="h-5 w-5 text-foreground" />
        </div>
        {!collapsed && (
          <span className="font-display text-[17px] tracking-tight text-foreground animate-fade-in">
            Veritas
          </span>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-6 space-y-1">
        {!collapsed && (
          <p className="px-3 mb-4 text-[10px] font-semibold uppercase tracking-[0.12em] text-muted/50">
            Workspace
          </p>
        )}
        {navigation.map((item) => {
          const isActive = pathname === item.href || pathname?.startsWith(item.href + "/")
          return (
            <Link
              key={item.name}
              href={item.href}
              className={cn(
                "flex items-center gap-3 rounded-sm px-3 py-2.5 text-[13px] font-medium transition-all duration-200",
                isActive
                  ? "bg-surface text-foreground"
                  : "text-muted hover:bg-surface/60 hover:text-foreground"
              )}
            >
              <item.icon className="h-[18px] w-[18px] shrink-0" />
              {!collapsed && <span>{item.name}</span>}
            </Link>
          )
        })}
      </nav>

      {/* Footer */}
      <div className="border-t border-border px-3 py-3 space-y-0.5">
        <button
          onClick={onToggle}
          className="flex w-full items-center gap-3 rounded-sm px-3 py-2.5 text-[13px] font-medium text-muted hover:bg-surface hover:text-foreground transition-colors cursor-pointer"
        >
          {collapsed ? (
            <ChevronRight className="h-[18px] w-[18px] shrink-0" />
          ) : (
            <>
              <ChevronLeft className="h-[18px] w-[18px] shrink-0" />
              <span>Collapse</span>
            </>
          )}
        </button>
        <button className="flex w-full items-center gap-3 rounded-sm px-3 py-2.5 text-[13px] font-medium text-muted hover:bg-surface hover:text-red-600 transition-colors cursor-pointer">
          <LogOut className="h-[18px] w-[18px] shrink-0" />
          {!collapsed && <span>Sign Out</span>}
        </button>
      </div>
    </aside>
  )
}
