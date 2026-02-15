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
    setTimeout(() => {
      localStorage.setItem("veritas_token", "demo-token")
      router.push("/dashboard")
    }, 800)
  }

  return (
    <div className="min-h-screen flex bg-primary">
      {/* Left — Brand storytelling */}
      <div className="hidden lg:flex lg:w-[55%] flex-col justify-between p-14 relative overflow-hidden">
        {/* Subtle gradient orbs — monochrome white glow */}
        <div className="absolute -bottom-40 -left-40 w-[500px] h-[500px] rounded-full bg-white/[0.03] blur-3xl" />
        <div className="absolute top-20 right-10 w-[300px] h-[300px] rounded-full bg-white/[0.02] blur-3xl" />

        <div className="flex items-center gap-3 relative z-10">
          <div className="flex h-9 w-9 items-center justify-center rounded-md bg-foreground">
            <Scale className="h-4 w-4 text-primary-foreground" />
          </div>
          <span className="font-display text-[17px] font-semibold text-primary-foreground tracking-tight">
            Veritas AI
          </span>
        </div>

        <div className="max-w-lg relative z-10">
          <h1 className="font-display text-[3.2rem] font-bold text-primary-foreground leading-[1.08] tracking-tight">
            Legal intelligence,<br />
            <span className="text-foreground">verified.</span>
          </h1>
          <p className="mt-5 text-[17px] text-white/40 leading-relaxed max-w-md">
            Every answer backed by source documents, page numbers,
            and confidence scores. Built for firms that demand precision.
          </p>

          {/* Proof points */}
          <div className="mt-10 grid grid-cols-3 gap-px bg-white/[0.06] rounded-lg overflow-hidden">
            {[
              { label: "Documents analyzed", value: "2.4M+" },
              { label: "Accuracy rate", value: "99.2%" },
              { label: "Firms onboarded", value: "340+" },
            ].map((stat) => (
              <div key={stat.label} className="bg-primary p-5">
                <p className="font-display text-xl font-semibold text-primary-foreground tracking-tight">{stat.value}</p>
                <p className="text-[11px] text-white/30 mt-1 uppercase tracking-[0.06em]">{stat.label}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-8 text-[11px] text-white/20 uppercase tracking-[0.08em] relative z-10">
          <span>SOC 2 Type II</span>
          <span className="h-3 w-px bg-white/10" />
          <span>GDPR</span>
          <span className="h-3 w-px bg-white/10" />
          <span>ISO 27001</span>
          <span className="h-3 w-px bg-white/10" />
          <span>No AI Training</span>
        </div>
      </div>

      {/* Right — Login form */}
      <div className="flex-1 flex items-center justify-center p-8 bg-white lg:rounded-l-[2rem]">
        <div className="w-full max-w-[380px]">
          {/* Mobile logo */}
          <div className="flex items-center gap-3 mb-10 lg:hidden">
            <div className="flex h-9 w-9 items-center justify-center rounded-md bg-foreground">
              <Scale className="h-4 w-4 text-primary-foreground" />
            </div>
            <span className="font-display text-[17px] font-semibold text-foreground">Veritas AI</span>
          </div>

          <h2 className="font-display text-[22px] font-semibold text-foreground tracking-tight">Welcome back</h2>
          <p className="text-[14px] text-muted mt-1.5">Sign in to continue to your workspace</p>

          <form onSubmit={handleSubmit} className="mt-8 space-y-5">
            {error && (
              <div className="rounded-md bg-[#FDF0ED] border border-[#E8C4BA] text-[#B04028] text-[13px] px-4 py-3">
                {error}
              </div>
            )}

            <div>
              <label className="block text-[13px] font-medium text-foreground mb-2">
                Email
              </label>
              <Input
                type="email"
                placeholder="name@firm.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                className="h-11 text-[14px] focus-visible:ring-foreground/10 focus-visible:border-foreground/30"
              />
            </div>

            <div>
              <label className="block text-[13px] font-medium text-foreground mb-2">
                Password
              </label>
              <div className="relative">
                <Input
                  type={showPassword ? "text" : "password"}
                  placeholder="Enter your password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  className="h-11 pr-10 text-[14px] focus-visible:ring-foreground/10 focus-visible:border-foreground/30"
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

            <div className="flex items-center justify-between">
              <label className="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" className="rounded border-border accent-foreground h-3.5 w-3.5" />
                <span className="text-[13px] text-muted">Remember me</span>
              </label>
              <a href="#" className="text-[13px] text-foreground hover:text-foreground/70 font-medium transition-colors">
                Forgot password?
              </a>
            </div>

            <Button type="submit" size="lg" className="w-full" disabled={isLoading}>
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

          <p className="text-center text-[13px] text-muted mt-8">
            Don&apos;t have an account?{" "}
            <a href="#" className="text-foreground hover:text-foreground/70 font-medium transition-colors">
              Request access
            </a>
          </p>
        </div>
      </div>
    </div>
  )
}
