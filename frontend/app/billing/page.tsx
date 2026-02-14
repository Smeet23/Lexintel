"use client"

import React from "react"
import {
  CreditCard,
  Download,
  Zap,
  TrendingUp,
  Calendar,
  FileText,
  ArrowUpRight,
  CheckCircle2,
  Clock,
  AlertCircle,
} from "lucide-react"
import AppLayout from "@/layouts/AppLayout"
import PageHeader from "@/components/PageHeader"
import StatsCard from "@/components/StatsCard"
import DataTable from "@/components/DataTable"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { formatDate } from "@/lib/utils"

interface Invoice {
  id: string
  date: string
  amount: number
  status: string
  plan: string
  [key: string]: unknown
}

const mockInvoices: Invoice[] = [
  { id: "INV-2026-002", date: "2026-02-01", amount: 299, status: "paid", plan: "Professional" },
  { id: "INV-2026-001", date: "2026-01-01", amount: 299, status: "paid", plan: "Professional" },
  { id: "INV-2025-012", date: "2025-12-01", amount: 199, status: "paid", plan: "Starter" },
  { id: "INV-2025-011", date: "2025-11-01", amount: 199, status: "paid", plan: "Starter" },
]

const usageByMatter = [
  { matter: "Acme vs Global Corp", tokens: 8234, queries: 45, cost: 24.70 },
  { matter: "TechStart IP Review", tokens: 12890, queries: 67, cost: 38.67 },
  { matter: "Nordic Data Privacy Audit", tokens: 9870, queries: 41, cost: 29.61 },
  { matter: "Smith Estate Planning", tokens: 4120, queries: 23, cost: 12.36 },
  { matter: "Metro Construction Dispute", tokens: 6450, queries: 31, cost: 19.35 },
]

export default function BillingPage() {
  const tokenLimit = 100000
  const tokenUsed = 42390
  const tokenPercent = Math.round((tokenUsed / tokenLimit) * 100)

  const invoiceColumns = [
    {
      key: "id",
      header: "Invoice",
      render: (item: Invoice) => (
        <div className="flex items-center gap-2">
          <FileText className="h-4 w-4 text-muted" />
          <span className="font-medium text-sm">{item.id}</span>
        </div>
      ),
    },
    {
      key: "date",
      header: "Date",
      render: (item: Invoice) => <span className="text-sm text-muted">{formatDate(item.date)}</span>,
    },
    {
      key: "plan",
      header: "Plan",
      render: (item: Invoice) => <span className="text-sm">{item.plan}</span>,
    },
    {
      key: "amount",
      header: "Amount",
      render: (item: Invoice) => <span className="text-sm font-medium">${item.amount.toFixed(2)}</span>,
    },
    {
      key: "status",
      header: "Status",
      render: (item: Invoice) => (
        <Badge variant={item.status === "paid" ? "active" : item.status === "pending" ? "review" : "error"}>
          {item.status === "paid" && <CheckCircle2 className="h-3 w-3 mr-1" />}
          {item.status === "pending" && <Clock className="h-3 w-3 mr-1" />}
          {item.status.charAt(0).toUpperCase() + item.status.slice(1)}
        </Badge>
      ),
    },
    {
      key: "actions",
      header: "",
      className: "w-12",
      render: () => (
        <Button variant="ghost" size="sm" className="h-8 text-xs text-accent">
          <Download className="h-3 w-3 mr-1" />
          PDF
        </Button>
      ),
    },
  ]

  return (
    <AppLayout title="Billing">
      <PageHeader
        title="Billing & Usage"
        description="Monitor token usage, manage your plan, and download invoices"
        actions={
          <Button variant="outline">
            <ArrowUpRight className="h-4 w-4" />
            Upgrade Plan
          </Button>
        }
      />

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <StatsCard title="Current Plan" value="Professional" icon={CreditCard} />
        <StatsCard title="Monthly Cost" value="$299" icon={CreditCard} />
        <StatsCard title="Tokens Used" value="42,390" icon={Zap} trend={{ value: "12%", positive: true }} />
        <StatsCard title="Queries This Month" value="156" icon={TrendingUp} trend={{ value: "18%", positive: true }} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        {/* Token Usage Bar */}
        <div className="lg:col-span-2 bg-white rounded-xl border border-border shadow-sm p-6">
          <h3 className="text-lg font-semibold text-foreground mb-4">Token Usage</h3>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-muted">Monthly usage</span>
                <span className="font-medium">{tokenUsed.toLocaleString()} / {tokenLimit.toLocaleString()}</span>
              </div>
              <Progress value={tokenPercent} />
              <p className="text-xs text-muted mt-1">{tokenPercent}% of monthly limit used</p>
            </div>

            {/* Per-matter breakdown */}
            <div className="border-t border-border pt-4">
              <h4 className="text-sm font-medium text-foreground mb-3">Usage by Matter</h4>
              <div className="space-y-3">
                {usageByMatter.map((item) => (
                  <div key={item.matter} className="flex items-center gap-3">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-sm text-foreground truncate">{item.matter}</span>
                        <span className="text-xs font-mono text-muted ml-2">{item.tokens.toLocaleString()}</span>
                      </div>
                      <div className="h-1.5 rounded-full bg-surface overflow-hidden">
                        <div
                          className="h-full rounded-full bg-accent transition-all"
                          style={{ width: `${(item.tokens / tokenUsed) * 100}%` }}
                        />
                      </div>
                    </div>
                    <span className="text-xs text-muted w-16 text-right">${item.cost.toFixed(2)}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Plan Details */}
        <div className="bg-white rounded-xl border border-border shadow-sm p-6">
          <h3 className="text-lg font-semibold text-foreground mb-4">Plan Details</h3>
          <div className="space-y-4">
            <div className="rounded-lg bg-accent/5 border border-accent/20 p-4">
              <p className="text-sm font-semibold text-accent">Professional Plan</p>
              <p className="text-2xl font-bold text-foreground mt-1">$299<span className="text-sm font-normal text-muted">/month</span></p>
            </div>
            <ul className="space-y-2.5">
              {[
                "100,000 tokens / month",
                "Unlimited matters",
                "5 team members",
                "Contract review",
                "Draft assistant",
                "Priority support",
              ].map((feature) => (
                <li key={feature} className="flex items-center gap-2 text-sm text-foreground">
                  <CheckCircle2 className="h-4 w-4 text-emerald-500 shrink-0" />
                  {feature}
                </li>
              ))}
            </ul>
            <div className="border-t border-border pt-4">
              <p className="text-xs text-muted">
                <Calendar className="inline h-3 w-3 mr-1" />
                Next billing date: March 1, 2026
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Invoices */}
      <div className="bg-white rounded-xl border border-border shadow-sm">
        <div className="p-4 border-b border-border flex items-center justify-between">
          <h3 className="font-semibold text-foreground">Invoices</h3>
          <Button variant="ghost" size="sm" className="text-accent">
            <Download className="h-4 w-4 mr-1" />
            Download All
          </Button>
        </div>
        <DataTable columns={invoiceColumns} data={mockInvoices} />
      </div>
    </AppLayout>
  )
}
