"use client"

import React, { useState, useEffect, useRef } from "react"
import Link from "next/link"
import { motion, useScroll, useTransform, useInView } from "framer-motion"
import {
  Scale,
  ArrowRight,
  FileText,
  MessageSquare,
  Shield,
  BookOpen,
  Brain,
  Search,
  Lock,
  ListChecks,
  ArrowUpRight,
  ChevronRight,
  Menu,
  X,
  Home,
  Send,
  Paperclip,
  Settings,
} from "lucide-react"
import { Button } from "@/components/ui/button"

const ease = [0.16, 1, 0.3, 1] as const

function FadeIn({
  children,
  delay = 0,
  direction = "up",
  className = "",
}: {
  children: React.ReactNode
  delay?: number
  direction?: "up" | "down" | "left" | "right" | "none"
  className?: string
}) {
  const ref = useRef(null)
  const isInView = useInView(ref, { once: true, margin: "-60px" })

  const directionMap = {
    up: { y: 24, x: 0 },
    down: { y: -24, x: 0 },
    left: { y: 0, x: -24 },
    right: { y: 0, x: 24 },
    none: { y: 0, x: 0 },
  }

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, ...directionMap[direction] }}
      animate={isInView ? { opacity: 1, y: 0, x: 0 } : {}}
      transition={{ duration: 0.7, delay, ease }}
      className={className}
    >
      {children}
    </motion.div>
  )
}

function Navbar() {
  const [scrolled, setScrolled] = useState(false)
  const [mobileOpen, setMobileOpen] = useState(false)

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20)
    window.addEventListener("scroll", onScroll)
    return () => window.removeEventListener("scroll", onScroll)
  }, [])

  return (
    <motion.header
      initial={{ y: -20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.6, ease }}
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-500 ${
        scrolled
          ? "glass border-b border-border/60 shadow-elevated"
          : "bg-transparent"
      }`}
    >
      <nav className="max-w-[1200px] mx-auto flex items-center justify-between px-6 h-16">
        <Link href="/" className="flex items-center gap-2.5 group">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-foreground transition-transform duration-300 group-hover:scale-105">
            <Scale className="h-4 w-4 text-primary-foreground" />
          </div>
          <span className="font-display text-[19px] text-foreground tracking-tight">
            LexIntel
          </span>
        </Link>

        {/* Desktop nav */}
        <div className="hidden md:flex items-center gap-8">
          {["Platform", "Solutions", "Security", "Resources"].map((item, i) => (
            <a
              key={item}
              href={`#${["platform", "features", "security", "faq"][i]}`}
              className="text-[14px] text-muted hover:text-foreground transition-colors relative after:absolute after:bottom-0 after:left-0 after:w-0 after:h-px after:bg-foreground after:transition-all hover:after:w-full"
            >
              {item}
            </a>
          ))}
        </div>

        <div className="hidden md:flex items-center gap-3">
          <Link href="/dashboard">
            <Button variant="ghost" size="sm" className="text-[14px]">
              Login
            </Button>
          </Link>
          <Link href="/dashboard">
            <Button size="sm" className="text-[13px] rounded-lg">
              Request a Demo
            </Button>
          </Link>
        </div>

        {/* Mobile menu button */}
        <button
          onClick={() => setMobileOpen(!mobileOpen)}
          className="md:hidden text-foreground"
        >
          {mobileOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </button>
      </nav>

      {/* Mobile menu */}
      {mobileOpen && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="md:hidden glass border-b border-border px-6 py-6 space-y-4"
        >
          <a href="#platform" className="block text-[15px] text-foreground" onClick={() => setMobileOpen(false)}>Platform</a>
          <a href="#features" className="block text-[15px] text-foreground" onClick={() => setMobileOpen(false)}>Solutions</a>
          <a href="#security" className="block text-[15px] text-foreground" onClick={() => setMobileOpen(false)}>Security</a>
          <a href="#faq" className="block text-[15px] text-foreground" onClick={() => setMobileOpen(false)}>Resources</a>
          <div className="pt-4 border-t border-border">
            <Link href="/dashboard">
              <Button className="w-full rounded-lg">Enter Platform</Button>
            </Link>
          </div>
        </motion.div>
      )}
    </motion.header>
  )
}

function HeroSection() {
  const ref = useRef(null)
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ["start start", "end start"],
  })
  const y = useTransform(scrollYProgress, [0, 1], [0, 100])
  const opacity = useTransform(scrollYProgress, [0, 0.8], [1, 0])

  return (
    <section ref={ref} className="relative pt-40 pb-28 px-6 overflow-hidden">
      {/* Ambient gradient background */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[-20%] right-[-10%] w-[700px] h-[700px] rounded-full bg-gradient-to-br from-surface via-surface-hover to-transparent blur-[100px] opacity-80" />
        <div className="absolute bottom-[-10%] left-[-5%] w-[500px] h-[500px] rounded-full bg-surface blur-[120px] opacity-50" />
        {/* Subtle grid pattern */}
        <div className="absolute inset-0 opacity-[0.03] grid-pattern-light" />
      </div>

      <motion.div style={{ y, opacity }} className="max-w-[1200px] mx-auto relative">
        <div className="max-w-3xl">
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1, ease }}
            className="text-[13px] font-medium text-muted uppercase tracking-[0.12em] mb-6"
          >
            Legal Intelligence Platform
          </motion.p>
          <motion.h1
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2, ease }}
            className="font-display text-[3.5rem] md:text-[4.5rem] lg:text-[5.5rem] leading-[1.02] tracking-tight text-foreground"
          >
            Engineered for<br />
            <span className="text-gradient">Every Task</span>
          </motion.h1>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.4, ease }}
            className="mt-8 text-[18px] md:text-[20px] text-muted leading-relaxed max-w-xl"
          >
            Every answer backed by source documents, page numbers, and confidence scores.
          </motion.p>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.55, ease }}
            className="mt-10 flex flex-col sm:flex-row gap-4"
          >
            <Link href="/dashboard">
              <Button size="lg" className="text-[15px] h-12 px-8 rounded-lg group">
                Enter Platform
                <ArrowRight className="h-4 w-4 ml-1 transition-transform group-hover:translate-x-0.5" />
              </Button>
            </Link>
            <a href="#platform">
              <Button variant="outline" size="lg" className="text-[15px] h-12 px-8 rounded-lg">
                Learn More
              </Button>
            </a>
          </motion.div>
        </div>
      </motion.div>
    </section>
  )
}

function ProductShowcase() {
  return (
    <section className="px-6 pb-24 -mt-4">
      <div className="max-w-[1200px] mx-auto">
        <div className="grid grid-cols-1 lg:grid-cols-[380px_1fr] gap-12 items-start">
          <FadeIn delay={0.1}>
            <div className="pt-8">
              <p className="text-[15px] text-muted leading-relaxed">
                LexIntel helps you search, review, and work with your legal documents.
              </p>
            </div>
          </FadeIn>

          <FadeIn delay={0.2}>
            <div className="bg-white rounded-xl border border-border shadow-elevated-lg overflow-hidden transition-shadow duration-500 hover:shadow-glow">
              <div className="flex h-[520px]">
                {/* Sidebar */}
                <div className="w-[190px] bg-surface/80 border-r border-border p-4 hidden md:flex flex-col">
                  <div className="flex items-center gap-2 mb-6">
                    <div className="flex h-6 w-6 items-center justify-center rounded-md bg-foreground">
                      <Scale className="h-3 w-3 text-primary-foreground" />
                    </div>
                    <span className="font-display text-[15px] text-foreground">LexIntel</span>
                  </div>

                  <nav className="space-y-1 text-[13px]">
                    <div className="flex items-center gap-2.5 px-2.5 py-2 rounded-md bg-white text-foreground font-medium shadow-sm">
                      <Home className="h-4 w-4" /> Dashboard
                    </div>
                    {[
                      { icon: FileText, label: "Matters" },
                      { icon: BookOpen, label: "Precedents" },
                      { icon: Search, label: "Team" },
                      { icon: Settings, label: "Settings" },
                    ].map((item) => (
                      <div key={item.label} className="flex items-center gap-2.5 px-2.5 py-2 rounded-md text-muted">
                        <item.icon className="h-4 w-4" /> {item.label}
                      </div>
                    ))}
                  </nav>

                  <div className="mt-6 pt-4 border-t border-border">
                    <p className="text-[10px] font-medium text-muted/50 uppercase tracking-[0.1em] px-2.5 mb-2">Recent Matters</p>
                    <div className="space-y-1.5 text-[12px] text-muted pl-2.5">
                      <p className="truncate">Acme vs Global Corp</p>
                      <p className="truncate">Smith Estate Planning</p>
                      <p className="truncate">TechStart IP Review</p>
                      <p className="truncate">Nordic Data Privacy</p>
                    </div>
                  </div>
                </div>

                {/* Main content area */}
                <div className="flex-1 flex flex-col items-center justify-center p-8 bg-white">
                  <h2 className="font-display text-[28px] text-foreground mb-4">LexIntel</h2>
                  <div className="flex items-center gap-4 mb-10">
                    {[
                      { icon: MessageSquare, label: "Ask AI" },
                      { icon: FileText, label: "Draft document" },
                      { icon: Shield, label: "Review contract" },
                    ].map((item) => (
                      <button key={item.label} className="flex items-center gap-1.5 text-[13px] text-muted hover:text-foreground transition-colors">
                        <item.icon className="h-3.5 w-3.5" /> {item.label}
                      </button>
                    ))}
                  </div>

                  <div className="w-full max-w-xl">
                    <div className="border border-border rounded-xl bg-white shadow-elevated">
                      <div className="px-4 py-4">
                        <p className="text-[14px] text-muted/50">Ask a question about your documents...</p>
                      </div>
                      <div className="flex items-center gap-3 px-4 py-3 border-t border-border">
                        <button className="flex items-center gap-1.5 text-[12px] text-muted">
                          <Paperclip className="h-3.5 w-3.5" /> Upload files
                        </button>
                        <button className="flex items-center gap-1.5 text-[12px] text-muted">
                          <BookOpen className="h-3.5 w-3.5" /> Citations
                        </button>
                        <div className="ml-auto">
                          <button className="h-7 w-7 rounded-md bg-foreground flex items-center justify-center text-white">
                            <Send className="h-3.5 w-3.5" />
                          </button>
                        </div>
                      </div>
                    </div>

                    <p className="text-center text-[11px] text-muted/50 mt-3">
                      Every answer includes source citations and confidence scores
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </FadeIn>
        </div>
      </div>
    </section>
  )
}

function PlatformSection() {
  const capabilities = [
    {
      icon: MessageSquare,
      title: "AI-Powered Research",
      description: "Ask natural language questions across your entire document corpus. Get precise answers with inline citations and confidence scores.",
    },
    {
      icon: FileText,
      title: "Contract Review",
      description: "Automated risk analysis identifies high-risk clauses, missing provisions, and compliance gaps across any agreement.",
    },
    {
      icon: Brain,
      title: "Draft Assistant",
      description: "Generate legal documents with inline source references. Every assertion traced back to your uploaded authorities.",
    },
    {
      icon: BookOpen,
      title: "Precedent Library",
      description: "Build a firm-wide knowledge base of approved outputs, research memos, and analytical frameworks.",
    },
    {
      icon: Search,
      title: "Document Intelligence",
      description: "Process PDFs, DOCX, and text files with automatic indexing and semantic search.",
    },
    {
      icon: ListChecks,
      title: "Audit Trail",
      description: "Complete transparency into every query, response, and citation. Full audit logging for compliance and quality control.",
    },
  ]

  return (
    <section id="platform" className="py-32 px-6">
      <div className="max-w-[1200px] mx-auto">
        <FadeIn>
          <div className="max-w-2xl mb-16">
            <p className="text-[12px] font-medium text-muted uppercase tracking-[0.12em] mb-4">
              Platform
            </p>
            <h2 className="font-display text-[2.5rem] md:text-[3rem] text-foreground tracking-tight leading-[1.1]">
              Built for the complexity<br />
              of legal work
            </h2>
            <p className="mt-5 text-[16px] text-muted leading-relaxed">
              A comprehensive platform that combines AI research, contract analysis,
              and document drafting — all with verifiable citations.
            </p>
          </div>
        </FadeIn>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          {capabilities.map((cap, i) => (
            <FadeIn key={cap.title} delay={i * 0.08}>
              <div className="bg-white rounded-xl border border-border p-8 md:p-9 group hover:shadow-elevated-lg hover:border-border-strong transition-all duration-400">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-surface group-hover:bg-surface-hover transition-colors">
                  <cap.icon className="h-5 w-5 text-foreground" strokeWidth={1.5} />
                </div>
                <h3 className="font-display text-[18px] text-foreground mt-5 mb-3">
                  {cap.title}
                </h3>
                <p className="text-[14px] text-muted leading-relaxed">
                  {cap.description}
                </p>
                <div className="mt-6 flex items-center gap-1 text-[13px] font-medium text-foreground opacity-0 group-hover:opacity-100 transition-all duration-300 translate-y-1 group-hover:translate-y-0">
                  Learn more <ChevronRight className="h-3.5 w-3.5" />
                </div>
              </div>
            </FadeIn>
          ))}
        </div>
      </div>
    </section>
  )
}

function FeaturesSection() {
  const solutions = [
    {
      title: "Litigation",
      description: "Reduce manual effort, prioritize strategy, and drive stronger outcomes in litigation.",
    },
    {
      title: "Transactional",
      description: "Accelerate due diligence, contract analysis, and review with precision and control.",
    },
    {
      title: "In-House",
      description: "Streamline work and shift focus to strategy and speed for legal teams.",
    },
    {
      title: "Regulatory",
      description: "Navigate complex regulatory landscapes with compliance monitoring.",
    },
    {
      title: "Knowledge Management",
      description: "Put your institutional knowledge to work across every matter and team.",
    },
  ]

  return (
    <section id="features" className="py-32 px-6 bg-primary text-white">
      <div className="max-w-[1200px] mx-auto">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-start">
          <FadeIn direction="right">
            <div>
              <p className="text-[12px] font-medium text-white/40 uppercase tracking-[0.12em] mb-4">
                Solutions
              </p>
              <h2 className="font-display text-[2.5rem] md:text-[3rem] text-white tracking-tight leading-[1.1]">
                Purpose-built for<br />
                every practice area
              </h2>
              <p className="mt-5 text-[16px] text-white/50 leading-relaxed max-w-md">
                Whether you are litigating, closing deals, or advising on regulatory matters —
                LexIntel adapts to how your team works.
              </p>
              <Link href="/dashboard" className="mt-8 inline-block">
                <Button
                  variant="outline"
                  size="lg"
                  className="text-[14px] bg-transparent border-white/20 text-white hover:bg-white/10 hover:text-white rounded-lg"
                >
                  Explore Platform
                  <ArrowRight className="h-4 w-4 ml-1" />
                </Button>
              </Link>
            </div>
          </FadeIn>

          <FadeIn direction="left" delay={0.15}>
            <div className="space-y-0 border-t border-white/10">
              {solutions.map((s) => (
                <div
                  key={s.title}
                  className="py-6 border-b border-white/10 group cursor-pointer"
                >
                  <div className="flex items-center justify-between">
                    <h3 className="text-[17px] font-medium text-white group-hover:text-white/80 transition-colors">
                      {s.title}
                    </h3>
                    <ArrowUpRight className="h-4 w-4 text-white/30 group-hover:text-white/70 transition-all group-hover:-translate-y-0.5 group-hover:translate-x-0.5" />
                  </div>
                  <p className="text-[14px] text-white/40 mt-2 leading-relaxed max-w-md group-hover:text-white/50 transition-colors">
                    {s.description}
                  </p>
                </div>
              ))}
            </div>
          </FadeIn>
        </div>
      </div>
    </section>
  )
}

function SecuritySection() {
  const features = [
    {
      icon: Lock,
      title: "Matter-scoped documents",
      description: "Your documents stay scoped to your matters.",
    },
  ]

  return (
    <section id="security" className="py-32 px-6">
      <div className="max-w-[1200px] mx-auto">
        <FadeIn>
          <div className="max-w-2xl mb-16">
            <p className="text-[12px] font-medium text-muted uppercase tracking-[0.12em] mb-4">Security</p>
            <h2 className="font-display text-[2.5rem] md:text-[3rem] text-foreground tracking-tight leading-[1.1]">Designed for<br />Sensitive Matters</h2>
          </div>
        </FadeIn>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {features.map((f, i) => (
            <FadeIn key={f.title} delay={i * 0.1}>
              <div className="bg-white border border-border rounded-xl p-8 hover:shadow-elevated-lg hover:border-border-strong transition-all duration-400 group">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-surface group-hover:bg-surface-hover transition-colors">
                  <f.icon className="h-5 w-5 text-foreground" strokeWidth={1.5} />
                </div>
                <h3 className="font-display text-[18px] text-foreground mt-5 mb-2">{f.title}</h3>
                <p className="text-[14px] text-muted leading-relaxed">{f.description}</p>
              </div>
            </FadeIn>
          ))}
        </div>
      </div>
    </section>
  )
}

function FAQSection() {
  const faqs = [
    {
      q: "How are answers supported?",
      a: "Answers can include references to source documents, page numbers, and confidence scores to help you review the supporting material.",
    },
    {
      q: "Where is my data hosted and processed?",
      a: "Your documents are associated with the matters you add them to. Contact us if you need details about infrastructure or data handling.",
    },
    {
      q: "Does LexIntel train on my data?",
      a: "Data handling depends on the service configuration. Review your applicable terms before adding sensitive documents.",
    },
    {
      q: "What file formats are supported?",
      a: "Available formats depend on the document processing service. Check the upload flow for the formats accepted by your workspace.",
    },
    {
      q: "How does access control work?",
      a: "Workspace access controls are still being developed. Avoid adding documents that require access restrictions until those controls are available.",
    },
  ]

  return (
    <section id="faq" className="py-32 px-6 bg-surface/60">
      <div className="max-w-[800px] mx-auto">
        <FadeIn>
          <p className="text-[12px] font-medium text-muted uppercase tracking-[0.12em] mb-4">
            Resources
          </p>
          <h2 className="font-display text-[2.5rem] md:text-[3rem] text-foreground tracking-tight leading-[1.1] mb-14">
            Frequently Asked<br />
            Questions
          </h2>
        </FadeIn>

        <div className="space-y-0">
          {faqs.map((faq, i) => (
            <FadeIn key={faq.q} delay={i * 0.06}>
              <details className="group border-b border-border">
                <summary className="flex items-center justify-between py-6 cursor-pointer list-none">
                  <h3 className="text-[16px] font-medium text-foreground pr-8">{faq.q}</h3>
                  <ChevronRight className="h-4 w-4 text-muted shrink-0 transition-transform duration-300 group-open:rotate-90" />
                </summary>
                <p className="text-[14px] text-muted leading-relaxed pb-6 pr-12">
                  {faq.a}
                </p>
              </details>
            </FadeIn>
          ))}
        </div>
      </div>
    </section>
  )
}

function CTASection() {
  return (
    <section className="py-32 px-6 bg-primary relative overflow-hidden">
      {/* Ambient glow */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[400px] rounded-full bg-white/[0.03] blur-[100px]" />

      <div className="max-w-[800px] mx-auto text-center relative">
        <FadeIn>
          <h2 className="font-display text-[2.5rem] md:text-[3.5rem] text-white tracking-tight leading-[1.08]">
            Ready to transform<br />
            your legal workflow?
          </h2>
          <p className="mt-6 text-[17px] text-white/40 leading-relaxed max-w-md mx-auto">
            Explore a workspace for searching, reviewing, and working with legal documents.
          </p>
          <div className="mt-10 flex flex-col sm:flex-row gap-4 justify-center">
            <Link href="/dashboard">
              <Button
                size="lg"
                className="bg-white text-foreground hover:bg-white/90 text-[15px] h-12 px-8 rounded-lg group"
              >
                Enter Platform
                <ArrowRight className="h-4 w-4 ml-1 transition-transform group-hover:translate-x-0.5" />
              </Button>
            </Link>
            <Button
              variant="outline"
              size="lg"
              className="bg-transparent border-white/20 text-white hover:bg-white/10 hover:text-white text-[15px] h-12 px-8 rounded-lg"
            >
              Request a Demo
            </Button>
          </div>
        </FadeIn>
      </div>
    </section>
  )
}

function Footer() {


  return (
    <footer className="border-t border-border bg-white py-16 px-6">
      <div className="max-w-[1200px] mx-auto">
        <div className="grid grid-cols-2 md:grid-cols-5 gap-10">
          <div className="col-span-2 md:col-span-1">
            <div className="flex items-center gap-2 mb-4">
              <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-foreground">
                <Scale className="h-3.5 w-3.5 text-primary-foreground" />
              </div>
              <span className="font-display text-[16px] text-foreground">LexIntel</span>
            </div>
            <p className="text-[13px] text-muted leading-relaxed">
              Legal intelligence,<br />verified.
            </p>
          </div>
    
        </div>

        <div className="mt-14 pt-6 border-t border-border flex flex-col sm:flex-row items-center justify-between gap-4">
          <p className="text-[12px] text-muted">
            &copy; 2026 LexIntel. All rights reserved.
          </p>

        </div>
      </div>
    </footer>
  )
}

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <HeroSection />
      <ProductShowcase />
      <PlatformSection />
      <FeaturesSection />
      <SecuritySection />
      <FAQSection />
      <CTASection />
      <Footer />
    </div>
  )
}
