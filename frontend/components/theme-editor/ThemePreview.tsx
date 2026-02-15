"use client"

import React, { useState } from "react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Switch } from "@/components/ui/switch"
import { Progress } from "@/components/ui/progress"
import { Briefcase, Bell, Search, Star, ArrowRight } from "lucide-react"

export default function ThemePreview() {
  const [activeTab, setActiveTab] = useState<"components" | "dashboard">("components")

  return (
    <div className="flex flex-col h-full">
      <div className="flex border-b border-border">
        <button
          onClick={() => setActiveTab("components")}
          className={`px-4 py-2.5 text-xs font-medium border-b-2 transition-colors ${
            activeTab === "components"
              ? "border-foreground text-foreground"
              : "border-transparent text-muted-foreground"
          }`}
        >
          Components
        </button>
        <button
          onClick={() => setActiveTab("dashboard")}
          className={`px-4 py-2.5 text-xs font-medium border-b-2 transition-colors ${
            activeTab === "dashboard"
              ? "border-foreground text-foreground"
              : "border-transparent text-muted-foreground"
          }`}
        >
          Mini Dashboard
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-6 space-y-8">
        {activeTab === "components" ? (
          <>
            {/* Buttons */}
            <section>
              <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">Buttons</h4>
              <div className="flex flex-wrap gap-2">
                <Button>Primary</Button>
                <Button variant="outline">Outline</Button>
                <Button variant="ghost">Ghost</Button>
                <Button variant="destructive">Destructive</Button>
              </div>
            </section>

            {/* Badges */}
            <section>
              <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">Badges</h4>
              <div className="flex flex-wrap gap-2">
                <Badge>Default</Badge>
                <Badge variant="active">Active</Badge>
                <Badge variant="review">Review</Badge>
                <Badge variant="closed">Closed</Badge>
              </div>
            </section>

            {/* Inputs */}
            <section>
              <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">Inputs</h4>
              <div className="space-y-3 max-w-sm">
                <Input placeholder="Search matters..." />
                <div className="flex items-center justify-between">
                  <span className="text-sm">Toggle option</span>
                  <Switch />
                </div>
              </div>
            </section>

            {/* Card */}
            <section>
              <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">Card</h4>
              <div className="rounded-xl border border-border bg-card p-6 shadow-elevated">
                <div className="flex items-center gap-3 mb-3">
                  <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-surface">
                    <Briefcase className="h-4 w-4 text-muted-foreground" />
                  </div>
                  <div>
                    <p className="text-sm font-medium text-card-foreground">Acme Corp Acquisition</p>
                    <p className="text-xs text-muted-foreground">PDF &middot; 3 days ago</p>
                  </div>
                </div>
                <Progress value={65} />
                <p className="text-xs text-muted-foreground mt-2">65% complete</p>
              </div>
            </section>

            {/* Typography */}
            <section>
              <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">Typography</h4>
              <div className="space-y-2">
                <h1 className="text-2xl font-display">Display Heading</h1>
                <h2 className="text-xl font-display">Section Heading</h2>
                <p className="text-sm text-foreground">Body text in the primary sans-serif font. This should be easy to read.</p>
                <p className="text-xs text-muted-foreground">Secondary text with muted foreground color.</p>
                <p className="text-xs font-mono text-muted-foreground">Monospace: const theme = &quot;active&quot;</p>
              </div>
            </section>

            {/* Colors palette */}
            <section>
              <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">Color Palette</h4>
              <div className="grid grid-cols-5 gap-2">
                {[
                  { name: "Primary", className: "bg-primary" },
                  { name: "Secondary", className: "bg-secondary" },
                  { name: "Accent", className: "bg-accent" },
                  { name: "Muted", className: "bg-muted" },
                  { name: "Surface", className: "bg-surface" },
                  { name: "Success", className: "bg-success" },
                  { name: "Warning", className: "bg-warning" },
                  { name: "Destructive", className: "bg-destructive" },
                  { name: "Border", className: "bg-border" },
                  { name: "Ring", className: "bg-ring" },
                ].map((c) => (
                  <div key={c.name} className="text-center">
                    <div className={`h-8 w-full rounded-md ${c.className}`} />
                    <p className="text-[10px] text-muted-foreground mt-1">{c.name}</p>
                  </div>
                ))}
              </div>
            </section>
          </>
        ) : (
          /* Mini Dashboard */
          <div className="space-y-4" style={{ transform: "scale(0.85)", transformOrigin: "top left" }}>
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-display font-semibold">Dashboard</h2>
              <Button size="sm"><Briefcase className="h-3 w-3" /> New Matter</Button>
            </div>

            <div className="grid grid-cols-3 gap-3">
              {[
                { title: "Active Matters", value: "12" },
                { title: "Processing", value: "3" },
                { title: "Queries Today", value: "47" },
              ].map((stat) => (
                <div key={stat.title} className="rounded-xl border border-border bg-card p-4 shadow-elevated">
                  <p className="text-[10px] uppercase tracking-wider text-muted-foreground">{stat.title}</p>
                  <p className="text-xl font-bold text-foreground mt-1">{stat.value}</p>
                </div>
              ))}
            </div>

            <div className="rounded-xl border border-border bg-card shadow-elevated">
              <div className="px-4 py-3 border-b border-border flex justify-between items-center">
                <h3 className="text-sm font-medium">Recent Matters</h3>
                <Button variant="ghost" size="sm" className="text-xs h-7">
                  View All <ArrowRight className="h-3 w-3 ml-1" />
                </Button>
              </div>
              {["Acme Corp Acquisition", "TechStart IP Review", "Smith Estate Planning"].map((name) => (
                <div key={name} className="px-4 py-3 border-b border-border last:border-0 flex items-center gap-3">
                  <div className="h-7 w-7 rounded-lg bg-surface flex items-center justify-center">
                    <Briefcase className="h-3 w-3 text-muted-foreground" />
                  </div>
                  <span className="text-xs text-foreground">{name}</span>
                  <Badge variant="active" className="ml-auto text-[10px]">Ready</Badge>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
