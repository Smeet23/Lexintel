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
        "fixed left-0 top-0 z-40 flex h-screen flex-col bg-primary text-white transition-all duration-300",
        collapsed ? "w-[72px]" : "w-64"
      )}
    >
      {/* Logo */}
      <div className="flex h-16 items-center gap-3 px-5 border-b border-white/10">
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-accent">
          <Scale className="h-5 w-5 text-white" />
        </div>
        {!collapsed && (
          <span className="text-lg font-semibold tracking-tight animate-fade-in">
            Veritas AI
          </span>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 space-y-1 px-3 py-4">
        {navigation.map((item) => {
          const isActive = pathname === item.href || pathname?.startsWith(item.href + "/")
          return (
            <Link
              key={item.name}
              href={item.href}
              className={cn(
                "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-all duration-200",
                isActive
                  ? "bg-white/15 text-white shadow-sm"
                  : "text-white/60 hover:bg-white/8 hover:text-white"
              )}
            >
              <item.icon className={cn("h-5 w-5 shrink-0", isActive && "text-accent")} />
              {!collapsed && <span className="animate-fade-in">{item.name}</span>}
            </Link>
          )
        })}
      </nav>

      {/* Footer */}
      <div className="border-t border-white/10 px-3 py-3 space-y-1">
        <button
          onClick={onToggle}
          className="flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium text-white/60 hover:bg-white/8 hover:text-white transition-colors cursor-pointer"
        >
          {collapsed ? (
            <ChevronRight className="h-5 w-5 shrink-0" />
          ) : (
            <>
              <ChevronLeft className="h-5 w-5 shrink-0" />
              <span>Collapse</span>
            </>
          )}
        </button>
        <button className="flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium text-white/60 hover:bg-white/8 hover:text-red-300 transition-colors cursor-pointer">
          <LogOut className="h-5 w-5 shrink-0" />
          {!collapsed && <span>Sign Out</span>}
        </button>
      </div>
    </aside>
  )
}
