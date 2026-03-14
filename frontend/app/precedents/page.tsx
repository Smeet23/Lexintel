"use client"

import React, { useState } from "react"
import {
  BookOpen,
  Search,
  Loader2,
  Trash2,
  Bookmark,
  FileText,
} from "lucide-react"
import AppLayout from "@/layouts/AppLayout"
import PageHeader from "@/components/PageHeader"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { cn, formatRelativeTime } from "@/lib/utils"
import {
  usePrecedentSearch,
  useSavedPrecedents,
  useSavePrecedent,
  useDeletePrecedent,
} from "@/hooks/use-matters"
import type { PrecedentSearchResult } from "@/lib/types"

export default function PrecedentsPage() {
  const [searchInput, setSearchInput] = useState("")
  const [searchQuery, setSearchQuery] = useState<string | null>(null)

  const searchResults = usePrecedentSearch(searchQuery)
  const savedPrecedents = useSavedPrecedents()
  const savePrecedent = useSavePrecedent()
  const deletePrecedent = useDeletePrecedent()

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    if (searchInput.trim().length >= 3) {
      setSearchQuery(searchInput.trim())
    }
  }

  const handleSave = (result: PrecedentSearchResult) => {
    savePrecedent.mutate({
      title: `${result.document_name} - ${result.section_name || "Page " + result.page_num}`,
      query: searchQuery || "",
      document_name: result.document_name,
      matter_id: result.matter_id,
      chunk_content: result.content,
      page_num: result.page_num ? parseInt(result.page_num) || null : null,
      section_name: result.section_name || null,
      relevance_score: String(result.relevance_score),
      tags: [],
      notes: null,
    })
  }

  return (
    <AppLayout title="Precedents">
      <PageHeader
        title="Precedent Library"
        description="Search across all matters for relevant legal precedents and save them for reference"
      />

      <Tabs defaultValue="search" className="space-y-6">
        <TabsList>
          <TabsTrigger value="search">
            <Search className="h-4 w-4 mr-1.5" />
            Search
          </TabsTrigger>
          <TabsTrigger value="saved">
            <Bookmark className="h-4 w-4 mr-1.5" />
            Saved
            {savedPrecedents.data && savedPrecedents.data.length > 0 && (
              <span className="ml-1.5 rounded-full bg-foreground/10 px-1.5 py-0.5 text-[10px] font-medium">
                {savedPrecedents.data.length}
              </span>
            )}
          </TabsTrigger>
        </TabsList>

        {/* Search Tab */}
        <TabsContent value="search">
          <form onSubmit={handleSearch} className="flex gap-3 mb-6">
            <div className="relative flex-1 max-w-xl">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted" />
              <Input
                placeholder="Search across all matters (e.g., 'indemnification clause', 'force majeure')..."
                value={searchInput}
                onChange={(e) => setSearchInput(e.target.value)}
                className="pl-9"
              />
            </div>
            <Button type="submit" disabled={searchInput.trim().length < 3 || searchResults.isFetching}>
              {searchResults.isFetching ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                "Search"
              )}
            </Button>
          </form>

          {/* Search results */}
          {!searchQuery && (
            <div className="text-center py-16 text-muted">
              <Search className="h-12 w-12 mx-auto mb-4 opacity-20" />
              <p className="text-sm">Enter a search query to find precedents across all your matters.</p>
              <p className="text-xs mt-1">Minimum 3 characters required.</p>
            </div>
          )}

          {searchQuery && searchResults.isLoading && (
            <div className="flex items-center justify-center py-16">
              <Loader2 className="h-6 w-6 animate-spin text-muted" />
              <span className="ml-3 text-sm text-muted">Searching across all matters...</span>
            </div>
          )}

          {searchQuery && !searchResults.isLoading && searchResults.data && (
            <>
              <p className="text-sm text-muted mb-4">
                {searchResults.data.total} result{searchResults.data.total !== 1 ? "s" : ""} for &ldquo;{searchQuery}&rdquo;
              </p>

              {searchResults.data.results.length === 0 && (
                <div className="text-center py-12 text-muted">
                  <p className="text-sm">No results found. Try a different search term.</p>
                </div>
              )}

              <div className="space-y-3">
                {searchResults.data.results.map((result, idx) => (
                  <div
                    key={idx}
                    className="bg-white rounded-xl border border-border shadow-sm p-5 hover:shadow-elevated transition-shadow"
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1.5 flex-wrap">
                          <Badge variant="default" className="text-xs">
                            {result.matter_name}
                          </Badge>
                          <span className="text-xs text-muted">
                            {result.document_name}
                            {result.page_num && `, p.${result.page_num}`}
                          </span>
                          <span className={cn(
                            "text-xs font-mono",
                            result.relevance_score >= 0.7 ? "text-emerald-600" :
                            result.relevance_score >= 0.4 ? "text-amber-600" : "text-muted"
                          )}>
                            {Math.round(result.relevance_score * 100)}% match
                          </span>
                        </div>

                        {result.section_name && (
                          <p className="text-sm font-medium text-foreground mb-1">{result.section_name}</p>
                        )}

                        <p className="text-sm text-muted line-clamp-3">
                          {result.content}
                        </p>
                      </div>

                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleSave(result)}
                        disabled={savePrecedent.isPending}
                        className="shrink-0"
                      >
                        <Bookmark className="h-4 w-4" />
                        Save
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
        </TabsContent>

        {/* Saved Tab */}
        <TabsContent value="saved">
          {savedPrecedents.isLoading && (
            <div className="flex items-center justify-center py-16">
              <Loader2 className="h-5 w-5 animate-spin text-muted" />
              <span className="ml-2 text-sm text-muted">Loading saved precedents...</span>
            </div>
          )}

          {!savedPrecedents.isLoading && (!savedPrecedents.data || savedPrecedents.data.length === 0) && (
            <div className="text-center py-16 text-muted">
              <BookOpen className="h-12 w-12 mx-auto mb-4 opacity-20" />
              <p className="text-sm">No saved precedents yet.</p>
              <p className="text-xs mt-1">Search for precedents and save them for future reference.</p>
            </div>
          )}

          {savedPrecedents.data && savedPrecedents.data.length > 0 && (
            <div className="space-y-3">
              {savedPrecedents.data.map((p) => (
                <div
                  key={p.id}
                  className="bg-white rounded-xl border border-border shadow-sm p-5"
                >
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1.5 flex-wrap">
                        <FileText className="h-4 w-4 text-muted shrink-0" />
                        <p className="text-sm font-medium text-foreground">{p.title}</p>
                      </div>

                      <div className="flex items-center gap-3 text-xs text-muted mb-2">
                        {p.document_name && <span>{p.document_name}</span>}
                        {p.page_num && <span>Page {p.page_num}</span>}
                        {p.relevance_score && (
                          <span className="font-mono">{Math.round(parseFloat(p.relevance_score) * 100)}% match</span>
                        )}
                        <span>{formatRelativeTime(p.created_at)}</span>
                      </div>

                      {p.chunk_content && (
                        <p className="text-sm text-muted line-clamp-2">{p.chunk_content}</p>
                      )}

                      {p.tags && p.tags.length > 0 && (
                        <div className="flex gap-1.5 mt-2">
                          {p.tags.map((tag) => (
                            <span key={tag} className="rounded-lg bg-surface px-2 py-0.5 text-xs text-muted">
                              {tag}
                            </span>
                          ))}
                        </div>
                      )}

                      {p.query && (
                        <p className="text-xs text-muted mt-2 italic">
                          Query: &ldquo;{p.query}&rdquo;
                        </p>
                      )}
                    </div>

                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => deletePrecedent.mutate(p.id)}
                      disabled={deletePrecedent.isPending}
                      className="shrink-0 text-muted hover:text-destructive"
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </TabsContent>
      </Tabs>
    </AppLayout>
  )
}
