"use client"

import React, { useState } from "react"
import { useRouter } from "next/navigation"
import { Scale, Eye, EyeOff, ArrowRight, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"

export default function LoginPage() {
  const router = useRouter()
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [showPassword, setShowPassword] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState("")

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError("")
    setIsLoading(true)

    // Demo mode - accept any credentials
    setTimeout(() => {
      localStorage.setItem("veritas_token", "demo-token")
      router.push("/dashboard")
    }, 800)
  }

  return (
    <div className="min-h-screen flex bg-primary">
      {/* Left side - Branding */}
      <div className="hidden lg:flex lg:w-1/2 flex-col justify-between p-12">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-accent">
            <Scale className="h-6 w-6 text-white" />
          </div>
          <span className="text-xl font-semibold text-white tracking-tight">Veritas AI</span>
        </div>

        <div className="max-w-md">
          <h1 className="text-4xl font-bold text-white leading-tight">
            Jurisdiction-Aware Legal AI with Verified Citations
          </h1>
          <p className="mt-4 text-lg text-white/60 leading-relaxed">
            Research smarter. Every answer backed by source documents, page numbers, and confidence scores.
          </p>

          <div className="mt-8 grid grid-cols-2 gap-4">
            {[
              { label: "Legal Research", desc: "AI-powered document analysis" },
              { label: "Contract Review", desc: "Clause-level risk analysis" },
              { label: "Audit Trail", desc: "Full provenance tracking" },
              { label: "Multi-Jurisdiction", desc: "US, UK, EU support" },
            ].map((feature) => (
              <div
                key={feature.label}
                className="rounded-lg border border-white/10 bg-white/5 p-4"
              >
                <p className="text-sm font-medium text-white">{feature.label}</p>
                <p className="text-xs text-white/50 mt-1">{feature.desc}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-6 text-xs text-white/40">
          <span>SOC 2 Compliant</span>
          <span>&middot;</span>
          <span>GDPR Ready</span>
          <span>&middot;</span>
          <span>No AI Training on Client Data</span>
        </div>
      </div>

      {/* Right side - Login Form */}
      <div className="flex-1 flex items-center justify-center p-8 bg-surface lg:rounded-l-3xl">
        <div className="w-full max-w-[400px]">
          {/* Mobile logo */}
          <div className="flex items-center gap-3 mb-8 lg:hidden">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-accent">
              <Scale className="h-6 w-6 text-white" />
            </div>
            <span className="text-xl font-semibold text-foreground">Veritas AI</span>
          </div>

          <h2 className="text-2xl font-semibold text-foreground">Welcome back</h2>
          <p className="text-sm text-muted mt-1">Sign in to your account to continue</p>

          <form onSubmit={handleSubmit} className="mt-8 space-y-4">
            {error && (
              <div className="rounded-lg bg-red-50 border border-red-200 text-red-700 text-sm px-4 py-3">
                {error}
              </div>
            )}

            <div>
              <label className="block text-sm font-medium text-foreground mb-1.5">
                Email address
              </label>
              <Input
                type="email"
                placeholder="name@firm.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                className="h-11"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-foreground mb-1.5">
                Password
              </label>
              <div className="relative">
                <Input
                  type={showPassword ? "text" : "password"}
                  placeholder="Enter your password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  className="h-11 pr-10"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted hover:text-foreground transition-colors cursor-pointer"
                >
                  {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
            </div>

            <div className="flex items-center justify-between text-sm">
              <label className="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" className="rounded border-border" />
                <span className="text-muted">Remember me</span>
              </label>
              <a href="#" className="text-accent hover:text-accent-hover font-medium">
                Forgot password?
              </a>
            </div>

            <Button type="submit" className="w-full h-11" disabled={isLoading}>
              {isLoading ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Signing in...
                </>
              ) : (
                <>
                  Sign In
                  <ArrowRight className="h-4 w-4" />
                </>
              )}
            </Button>
          </form>

          <p className="text-center text-sm text-muted mt-6">
            Don&apos;t have an account?{" "}
            <a href="#" className="text-accent hover:text-accent-hover font-medium">
              Request access
            </a>
          </p>
        </div>
      </div>
    </div>
  )
}
