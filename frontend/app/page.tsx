"use client"

import React, { useState, useEffect } from "react"
import Link from "next/link"
import {
  Scale,
  ArrowRight,
  FileText,
  MessageSquare,
  Shield,
  BookOpen,
  Brain,
  Search,
  CheckCircle2,
  Lock,
  Eye,
  XCircle,
  ListChecks,
  ShieldCheck,
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

function Navbar() {
  const [scrolled, setScrolled] = useState(false)
  const [mobileOpen, setMobileOpen] = useState(false)

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20)
    window.addEventListener("scroll", onScroll)
    return () => window.removeEventListener("scroll", onScroll)
  }, [])

  return (
    <header
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        scrolled
          ? "bg-white/90 backdrop-blur-md border-b border-border"
          : "bg-transparent"
      }`}
    >
      <nav className="max-w-[1200px] mx-auto flex items-center justify-between px-6 h-16">
        <Link href="/" className="flex items-center gap-2.5">
          <Scale className="h-5 w-5 text-foreground" />
          <span className="font-display text-[19px] text-foreground tracking-tight">
            LexIntel
          </span>
        </Link>

        {/* Desktop nav */}
        <div className="hidden md:flex items-center gap-8">
          <a href="#platform" className="text-[14px] text-muted hover:text-foreground transition-colors">
            Platform
          </a>
          <a href="#features" className="text-[14px] text-muted hover:text-foreground transition-colors">
            Solutions
          </a>
          <a href="#security" className="text-[14px] text-muted hover:text-foreground transition-colors">
            Security
          </a>
          <a href="#faq" className="text-[14px] text-muted hover:text-foreground transition-colors">
            Resources
          </a>
        </div>

        <div className="hidden md:flex items-center gap-3">
          <Link href="/dashboard">
            <Button variant="ghost" size="sm" className="text-[14px]">
              Login
            </Button>
          </Link>
          <Link href="/dashboard">
            <Button size="sm" className="text-[13px]">
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
        <div className="md:hidden bg-white border-b border-border px-6 py-6 space-y-4">
          <a href="#platform" className="block text-[15px] text-foreground" onClick={() => setMobileOpen(false)}>Platform</a>
          <a href="#features" className="block text-[15px] text-foreground" onClick={() => setMobileOpen(false)}>Solutions</a>
          <a href="#security" className="block text-[15px] text-foreground" onClick={() => setMobileOpen(false)}>Security</a>
          <a href="#faq" className="block text-[15px] text-foreground" onClick={() => setMobileOpen(false)}>Resources</a>
          <div className="pt-4 border-t border-border">
            <Link href="/dashboard">
              <Button className="w-full">Enter Platform</Button>
            </Link>
          </div>
        </div>
      )}
    </header>
  )
}

function HeroSection() {
  return (
    <section className="relative pt-40 pb-24 px-6 overflow-hidden">
      {/* Subtle background element */}
      <div className="absolute top-0 right-0 w-[600px] h-[600px] rounded-full bg-surface blur-[120px] opacity-60" />

      <div className="max-w-[1200px] mx-auto relative">
        <div className="max-w-3xl">
          <p className="text-[13px] font-medium text-muted uppercase tracking-[0.1em] mb-6">
            Legal Intelligence Platform
          </p>
          <h1 className="font-display text-[3.5rem] md:text-[4.5rem] lg:text-[5.5rem] leading-[1.04] tracking-tight text-foreground">
            Engineered for<br />
            Every Task
          </h1>
          <p className="mt-8 text-[18px] md:text-[20px] text-muted leading-relaxed max-w-xl">
            Trusted by legal professionals worldwide. Every answer backed by
            source documents, page numbers, and confidence scores.
          </p>
          <div className="mt-10 flex flex-col sm:flex-row gap-4">
            <Link href="/dashboard">
              <Button size="lg" className="text-[15px] h-12 px-8">
                Enter Platform
                <ArrowRight className="h-4 w-4 ml-1" />
              </Button>
            </Link>
            <a href="#platform">
              <Button variant="outline" size="lg" className="text-[15px] h-12 px-8">
                Learn More
              </Button>
            </a>
          </div>
        </div>
      </div>
    </section>
  )
}

function ProductShowcase() {
  return (
    <section className="px-6 pb-20 -mt-4">
      <div className="max-w-[1200px] mx-auto">
        {/* Left text + Right product mockup like Harvey */}
        <div className="grid grid-cols-1 lg:grid-cols-[380px_1fr] gap-12 items-start">
          <div className="pt-8">
            <p className="text-[15px] text-muted leading-relaxed">
              More than <span className="text-foreground font-medium">10,000 legal professionals</span> use
              LexIntel to cut through complexity to navigate complex legal work.
            </p>
          </div>

          {/* Product mockup — shows actual LexIntel app UI */}
          <div className="bg-surface rounded-lg border border-border shadow-lg overflow-hidden">
            <div className="flex h-[520px]">
              {/* Sidebar — matches real app */}
              <div className="w-[190px] bg-surface border-r border-border p-4 hidden md:flex flex-col">
                <div className="flex items-center gap-2 mb-6">
                  <Scale className="h-4 w-4 text-foreground" />
                  <span className="font-display text-[15px] text-foreground">LexIntel</span>
                </div>

                <nav className="space-y-1 text-[13px]">
                  <div className="flex items-center gap-2.5 px-2.5 py-2 rounded-sm bg-white text-foreground font-medium shadow-sm">
                    <Home className="h-4 w-4" /> Dashboard
                  </div>
                  <div className="flex items-center gap-2.5 px-2.5 py-2 rounded-sm text-muted">
                    <FileText className="h-4 w-4" /> Matters
                  </div>
                  <div className="flex items-center gap-2.5 px-2.5 py-2 rounded-sm text-muted">
                    <BookOpen className="h-4 w-4" /> Precedents
                  </div>
                  <div className="flex items-center gap-2.5 px-2.5 py-2 rounded-sm text-muted">
                    <Search className="h-4 w-4" /> Team
                  </div>
                  <div className="flex items-center gap-2.5 px-2.5 py-2 rounded-sm text-muted">
                    <Settings className="h-4 w-4" /> Settings
                  </div>
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
                  <button className="flex items-center gap-1.5 text-[13px] text-muted hover:text-foreground transition-colors">
                    <MessageSquare className="h-3.5 w-3.5" /> Ask AI
                  </button>
                  <button className="flex items-center gap-1.5 text-[13px] text-muted hover:text-foreground transition-colors">
                    <FileText className="h-3.5 w-3.5" /> Draft document
                  </button>
                  <button className="flex items-center gap-1.5 text-[13px] text-muted hover:text-foreground transition-colors">
                    <Shield className="h-3.5 w-3.5" /> Review contract
                  </button>
                </div>

                {/* Chat input */}
                <div className="w-full max-w-xl">
                  <div className="border border-border rounded-lg bg-white shadow-sm">
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
                        <button className="h-7 w-7 rounded-sm bg-foreground flex items-center justify-center text-white">
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
        </div>
      </div>
    </section>
  )
}

function MetricsSection() {
  const metrics = [
    { value: "200K+", label: "Queries processed daily" },
    { value: "1.3M+", label: "Documents analyzed" },
    { value: "99.2%", label: "Citation accuracy" },
    { value: "92%", label: "Monthly active usage" },
  ]

  return (
    <section className="py-20 px-6 border-y border-border bg-surface">
      <div className="max-w-[1200px] mx-auto">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-12 md:gap-8">
          {metrics.map((m) => (
            <div key={m.label} className="text-center md:text-left">
              <p className="font-display text-[2.5rem] md:text-[3rem] text-foreground tracking-tight leading-none">
                {m.value}
              </p>
              <p className="text-[13px] text-muted mt-3 leading-snug">
                {m.label}
              </p>
            </div>
          ))}
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
      description: "Process PDFs, DOCX, and scanned documents. Automatic indexing, OCR, and semantic search across millions of pages.",
    },
    {
      icon: ListChecks,
      title: "Audit Trail",
      description: "Complete transparency into every query, response, and citation. Full audit logging for compliance and quality control.",
    },
  ]

  return (
    <section id="platform" className="py-28 px-6">
      <div className="max-w-[1200px] mx-auto">
        <div className="max-w-2xl mb-16">
          <p className="text-[12px] font-medium text-muted uppercase tracking-[0.1em] mb-4">
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

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-px bg-border rounded-sm overflow-hidden">
          {capabilities.map((cap) => (
            <div key={cap.title} className="bg-white p-8 md:p-10 group">
              <cap.icon className="h-5 w-5 text-foreground mb-5" strokeWidth={1.5} />
              <h3 className="font-display text-[18px] text-foreground mb-3">
                {cap.title}
              </h3>
              <p className="text-[14px] text-muted leading-relaxed">
                {cap.description}
              </p>
              <div className="mt-6 flex items-center gap-1 text-[13px] font-medium text-foreground opacity-0 group-hover:opacity-100 transition-opacity">
                Learn more <ChevronRight className="h-3.5 w-3.5" />
              </div>
            </div>
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
      description: "Navigate complex regulatory landscapes with AI-powered compliance monitoring.",
    },
    {
      title: "Knowledge Management",
      description: "Put your institutional knowledge to work across every matter and team.",
    },
  ]

  return (
    <section id="features" className="py-28 px-6 bg-primary text-white">
      <div className="max-w-[1200px] mx-auto">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-start">
          <div>
            <p className="text-[12px] font-medium text-white/40 uppercase tracking-[0.1em] mb-4">
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
                className="text-[14px] border-white/20 text-white hover:bg-white/10 hover:text-white"
              >
                Explore Platform
                <ArrowRight className="h-4 w-4 ml-1" />
              </Button>
            </Link>
          </div>

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
                  <ArrowUpRight className="h-4 w-4 text-white/30 group-hover:text-white/70 transition-colors" />
                </div>
                <p className="text-[14px] text-white/40 mt-2 leading-relaxed max-w-md">
                  {s.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}

function SecuritySection() {
  const features = [
    {
      icon: ShieldCheck,
      title: "Dedicated Security Expertise",
      description: "In-house security team spans infrastructure, product, and operations with 24/7 coverage and end-to-end monitoring.",
    },
    {
      icon: Lock,
      title: "Data Sovereignty and Control",
      description: "Retain full control over your data. Set retention policies, delete data anytime, and keep everything in-region.",
    },
    {
      icon: XCircle,
      title: "No Model Training",
      description: "We contractually guarantee your data stays yours. We never use inputs, outputs, or uploaded documents to train models.",
    },
    {
      icon: Eye,
      title: "Independently Tested",
      description: "Regular third-party penetration testing, SOC 2 Type II and ISO 27001 audits by top-tier security firms.",
    },
  ]

  const badges = ["SOC 2 Type II", "ISO 27001", "GDPR", "CCPA"]

  return (
    <section id="security" className="py-28 px-6">
      <div className="max-w-[1200px] mx-auto">
        <div className="max-w-2xl mb-16">
          <p className="text-[12px] font-medium text-muted uppercase tracking-[0.1em] mb-4">
            Security
          </p>
          <h2 className="font-display text-[2.5rem] md:text-[3rem] text-foreground tracking-tight leading-[1.1]">
            For the Most<br />
            Sensitive Matters
          </h2>
          <p className="mt-5 text-[16px] text-muted leading-relaxed">
            Built on a non-negotiable principle: protecting the security and confidentiality
            of your data. Enterprise-grade controls from the ground up.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-16">
          {features.map((f) => (
            <div
              key={f.title}
              className="bg-white border border-border rounded-sm p-8 hover:border-border-strong transition-colors"
            >
              <f.icon className="h-5 w-5 text-foreground mb-5" strokeWidth={1.5} />
              <h3 className="font-display text-[18px] text-foreground mb-2">
                {f.title}
              </h3>
              <p className="text-[14px] text-muted leading-relaxed">
                {f.description}
              </p>
            </div>
          ))}
        </div>

        {/* Compliance badges */}
        <div className="bg-surface rounded-sm p-10 md:p-14">
          <h3 className="font-display text-[20px] text-foreground mb-3">
            Enterprise-grade security and controls
          </h3>
          <p className="text-[14px] text-muted leading-relaxed max-w-lg mb-8">
            Our platform is designed to safeguard the most sensitive information,
            with certifications that meet the highest industry standards.
          </p>
          <div className="flex flex-wrap gap-3">
            {badges.map((badge) => (
              <div
                key={badge}
                className="flex items-center gap-2 bg-white border border-border rounded-sm px-5 py-3"
              >
                <CheckCircle2 className="h-4 w-4 text-foreground" />
                <span className="text-[14px] font-medium text-foreground">{badge}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}

function FAQSection() {
  const faqs = [
    {
      q: "How does LexIntel ensure citation accuracy?",
      a: "Every response includes direct references to source documents with page numbers, section identifiers, and relevance scores. Our retrieval system cross-validates against the original corpus to maintain 99%+ citation accuracy.",
    },
    {
      q: "Where is my data hosted and processed?",
      a: "Data is hosted on enterprise-grade cloud infrastructure with SOC 2 Type II certification. We offer regional data residency options including US, EU, and Australia.",
    },
    {
      q: "Does LexIntel train on my data?",
      a: "No. We contractually prohibit using customer data for model training. Your documents, queries, and responses are never used to improve or train any models.",
    },
    {
      q: "What file formats are supported?",
      a: "We support PDF, DOCX, TXT, and scanned documents with OCR. Our processing pipeline handles files up to 50MB with automatic page-level indexing.",
    },
    {
      q: "How does access control work?",
      a: "Role-based access controls ensure only authorized users can access specific matters. Partners, associates, and paralegals each have configurable permission levels.",
    },
  ]

  return (
    <section id="faq" className="py-28 px-6 bg-surface">
      <div className="max-w-[800px] mx-auto">
        <p className="text-[12px] font-medium text-muted uppercase tracking-[0.1em] mb-4">
          Resources
        </p>
        <h2 className="font-display text-[2.5rem] md:text-[3rem] text-foreground tracking-tight leading-[1.1] mb-14">
          Frequently Asked<br />
          Questions
        </h2>

        <div className="space-y-0">
          {faqs.map((faq) => (
            <details key={faq.q} className="group border-b border-border">
              <summary className="flex items-center justify-between py-6 cursor-pointer list-none">
                <h3 className="text-[16px] font-medium text-foreground pr-8">{faq.q}</h3>
                <ChevronRight className="h-4 w-4 text-muted shrink-0 transition-transform group-open:rotate-90" />
              </summary>
              <p className="text-[14px] text-muted leading-relaxed pb-6 pr-12">
                {faq.a}
              </p>
            </details>
          ))}
        </div>
      </div>
    </section>
  )
}

function CTASection() {
  return (
    <section className="py-28 px-6 bg-primary">
      <div className="max-w-[800px] mx-auto text-center">
        <h2 className="font-display text-[2.5rem] md:text-[3.5rem] text-white tracking-tight leading-[1.08]">
          Ready to transform<br />
          your legal workflow?
        </h2>
        <p className="mt-6 text-[17px] text-white/40 leading-relaxed max-w-md mx-auto">
          Join hundreds of firms using LexIntel to research,
          analyze, and draft with verifiable precision.
        </p>
        <div className="mt-10 flex flex-col sm:flex-row gap-4 justify-center">
          <Link href="/dashboard">
            <Button
              size="lg"
              className="bg-white text-foreground hover:bg-white/90 text-[15px] h-12 px-8"
            >
              Enter Platform
              <ArrowRight className="h-4 w-4 ml-1" />
            </Button>
          </Link>
          <Button
            variant="outline"
            size="lg"
            className="border-white/20 text-white hover:bg-white/10 hover:text-white text-[15px] h-12 px-8"
          >
            Request a Demo
          </Button>
        </div>
      </div>
    </section>
  )
}

function Footer() {
  const columns = [
    {
      title: "Platform",
      links: ["AI Research", "Contract Review", "Draft Assistant", "Document Intelligence", "Audit Trail"],
    },
    {
      title: "Solutions",
      links: ["Litigation", "Transactional", "In-House", "Regulatory", "Knowledge Management"],
    },
    {
      title: "Company",
      links: ["Security", "Customers", "About", "Careers", "Newsroom"],
    },
    {
      title: "Resources",
      links: ["Documentation", "Blog", "Guides", "Privacy Policy", "Terms of Service"],
    },
  ]

  return (
    <footer className="border-t border-border bg-white py-16 px-6">
      <div className="max-w-[1200px] mx-auto">
        <div className="grid grid-cols-2 md:grid-cols-5 gap-10">
          <div className="col-span-2 md:col-span-1">
            <div className="flex items-center gap-2 mb-4">
              <Scale className="h-4 w-4 text-foreground" />
              <span className="font-display text-[16px] text-foreground">LexIntel</span>
            </div>
            <p className="text-[13px] text-muted leading-relaxed">
              Legal intelligence,<br />verified.
            </p>
          </div>
          {columns.map((col) => (
            <div key={col.title}>
              <p className="text-[12px] font-medium text-foreground uppercase tracking-[0.06em] mb-4">
                {col.title}
              </p>
              <ul className="space-y-2.5">
                {col.links.map((link) => (
                  <li key={link}>
                    <a href="#" className="text-[13px] text-muted hover:text-foreground transition-colors">
                      {link}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        <div className="mt-14 pt-6 border-t border-border flex flex-col sm:flex-row items-center justify-between gap-4">
          <p className="text-[12px] text-muted">
            &copy; 2026 LexIntel. All rights reserved.
          </p>
          <div className="flex items-center gap-6 text-[12px] text-muted">
            <a href="#" className="hover:text-foreground transition-colors">Privacy</a>
            <a href="#" className="hover:text-foreground transition-colors">Terms</a>
            <a href="#" className="hover:text-foreground transition-colors">Security</a>
          </div>
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
      <MetricsSection />
      <PlatformSection />
      <FeaturesSection />
      <SecuritySection />
      <FAQSection />
      <CTASection />
      <Footer />
    </div>
  )
}
