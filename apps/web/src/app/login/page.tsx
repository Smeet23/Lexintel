"use client"

import React, { useState } from "react"
import { useRouter } from "next/navigation"
import { motion } from "framer-motion"
import { Scale, Eye, EyeOff, ArrowRight, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"

const ease = [0.16, 1, 0.3, 1] as const

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
      localStorage.setItem("lexintel_token", "demo-token")
      router.push("/dashboard")
    }, 800)
  }

  return (
    <div className="min-h-screen flex bg-primary">
      {/* Left — Brand storytelling */}
      <div className="hidden lg:flex lg:w-[55%] flex-col justify-between p-14 relative overflow-hidden">
        {/* Ambient gradient orbs */}
        <div className="absolute -bottom-40 -left-40 w-[500px] h-[500px] rounded-full bg-white/[0.03] blur-[100px]" />
        <div className="absolute top-20 right-10 w-[300px] h-[300px] rounded-full bg-white/[0.02] blur-[80px]" />
        {/* Subtle grid */}
        <div className="absolute inset-0 opacity-[0.02] grid-pattern-dark" />

        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease }}
          className="flex items-center gap-3 relative z-10"
        >
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-white/10 backdrop-blur-sm border border-white/10">
            <Scale className="h-4 w-4 text-primary-foreground" />
          </div>
          <span className="font-display text-[17px] font-semibold text-primary-foreground tracking-tight">
            LexIntel
          </span>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.15, ease }}
          className="max-w-lg relative z-10"
        >
          <h1 className="font-display text-[3.2rem] font-bold text-primary-foreground leading-[1.08] tracking-tight">
            Legal intelligence,<br />
            <span className="text-white/30">verified.</span>
          </h1>
          <p className="mt-5 text-[17px] text-white/40 leading-relaxed max-w-md">
            Every answer backed by source documents, page numbers,
            and confidence scores.
          </p>

        </motion.div>
      </div>

      {/* Right — Login form */}
      <div className="flex-1 flex items-center justify-center p-8 bg-white lg:rounded-l-[2rem]">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1, ease }}
          className="w-full max-w-[380px]"
        >
          {/* Mobile logo */}
          <div className="flex items-center gap-3 mb-10 lg:hidden">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-foreground">
              <Scale className="h-4 w-4 text-primary-foreground" />
            </div>
            <span className="font-display text-[17px] font-semibold text-foreground">LexIntel</span>
          </div>

          <h2 className="font-display text-[22px] font-semibold text-foreground tracking-tight">Welcome back</h2>
          <p className="text-[14px] text-muted mt-1.5">Sign in to continue to your workspace</p>

          <form onSubmit={handleSubmit} className="mt-8 space-y-5">
            {error && (
              <div className="rounded-lg bg-red-50 border border-red-200/60 text-red-700 text-[13px] px-4 py-3">
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
                className="h-11 text-[14px] rounded-lg focus-visible:ring-foreground/10 focus-visible:border-foreground/30"
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
                  className="h-11 pr-10 text-[14px] rounded-lg focus-visible:ring-foreground/10 focus-visible:border-foreground/30"
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

            </div>

            <Button type="submit" size="lg" className="w-full rounded-lg" disabled={isLoading}>
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
            <span className="text-muted">Request access</span>
          </p>
        </motion.div>
      </div>
    </div>
  )
}
