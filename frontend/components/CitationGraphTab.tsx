"use client"

import { useState } from "react"
import { Loader2, Network, RefreshCw, AlertTriangle, Info, X } from "lucide-react"
import CitationGraph from "@/components/CitationGraph"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { useMatterGraph } from "@/hooks/use-matters"
import type { GraphNode } from "@/lib/types"

interface CitationGraphTabProps {
  matterId: string
}

function goodLawBadge(node: GraphNode) {
  const status =
    node.good_law_status ??
    (node.is_good_law === false ? "bad" : node.is_good_law === true ? "good" : "unknown")
  if (status === "bad") return <Badge variant="error">Bad law</Badge>
  if (status === "good") return <Badge variant="active">Good law</Badge>
  if (status === "no_adverse") return <Badge variant="active">No adverse treatment</Badge>
  return <Badge variant="default">Not validated</Badge>
}

export default function CitationGraphTab({ matterId }: CitationGraphTabProps) {
  const { data, isLoading, isError, error, refetch, isFetching } = useMatterGraph(matterId)
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null)

  // Exclude provenance "document" nodes — they're the upload itself, not a
  // legal authority. Showing them produces a misleading doc-centred star and a
  // wrong "N cases" count. Keep only real authorities and the edges between
  // them (drops dangling document_cites edges whose source was the doc node).
  const allNodes = data?.nodes ?? []
  const nodes = allNodes.filter((n) => n.type !== "document")
  const visibleIds = new Set(nodes.map((n) => n.id))
  const edges = (data?.edges ?? []).filter(
    (e) => visibleIds.has(e.source) && visibleIds.has(e.target)
  )
  const treatmentComputed = nodes.some(
    (n) => n.is_good_law !== null && n.is_good_law !== undefined
  )

  return (
    <div className="bg-white rounded-xl border border-border shadow-elevated">
      <div className="p-4 border-b border-border flex items-center justify-between gap-3">
        <div className="flex items-center gap-2 min-w-0">
          <Network className="h-4 w-4 text-muted shrink-0" />
          <h3 className="font-display font-semibold text-foreground truncate">Citation Graph</h3>
          {data && (
            <span className="text-xs text-muted shrink-0">
              {/* Count what is actually RENDERED (document nodes + their
                  provenance edges are filtered out above), so the header never
                  claims more links than the canvas draws. */}
              {nodes.length} cases · {edges.length} links
            </span>
          )}
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => refetch()}
          disabled={isFetching}
          aria-label="Refresh citation graph"
        >
          <RefreshCw className={`h-4 w-4 ${isFetching ? "animate-spin" : ""}`} />
          <span className="hidden sm:inline ml-1.5">Refresh</span>
        </Button>
      </div>

      {isLoading && (
        <div className="flex items-center justify-center py-16">
          <Loader2 className="h-5 w-5 animate-spin text-muted" />
          <span className="ml-2 text-sm text-muted">Building citation graph...</span>
        </div>
      )}

      {isError && (
        <div className="text-center py-16 text-muted">
          <AlertTriangle className="h-10 w-10 mx-auto mb-3 opacity-40 text-amber-500" />
          <p className="text-sm">
            {error instanceof Error ? error.message : "Failed to load citation graph."}
          </p>
          <Button variant="outline" size="sm" className="mt-4" onClick={() => refetch()}>
            Try again
          </Button>
        </div>
      )}

      {!isLoading && !isError && nodes.length === 0 && (
        <div className="text-center py-16 text-muted">
          <Network className="h-10 w-10 mx-auto mb-3 opacity-30" />
          <p className="text-sm">No citation relationships found yet.</p>
          <p className="text-xs mt-1 max-w-md mx-auto">
            The citation graph is built from cases and statutes detected in your documents during
            processing. Upload documents that cite legal authorities to populate it.
          </p>
        </div>
      )}

      {!isLoading && !isError && nodes.length > 0 && (
        <div className="relative">
          {!treatmentComputed && (
            <div className="mx-4 mt-4 flex items-start gap-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
              <Info className="h-4 w-4 shrink-0 mt-0.5 text-amber-500" />
              <span>
                Good-law treatment not yet computed for these authorities — showing citation
                structure only.
              </span>
            </div>
          )}
          <CitationGraph
            graphNodes={nodes}
            graphEdges={edges}
            onNodeClick={(node) => setSelectedNode(node)}
          />

          {selectedNode && (
            <div className="absolute top-3 right-3 w-72 max-w-[80%] bg-white rounded-lg border border-border shadow-elevated p-4 text-sm">
              <div className="flex items-start justify-between gap-2">
                <p className="font-medium text-foreground break-words">
                  {selectedNode.name || selectedNode.citation}
                </p>
                <button
                  onClick={() => setSelectedNode(null)}
                  className="text-muted hover:text-foreground shrink-0"
                  aria-label="Close details"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
              <p className="text-xs text-muted mt-1 break-words">{selectedNode.citation}</p>
              <div className="flex flex-wrap gap-1.5 mt-3">
                <Badge variant="default">{selectedNode.type}</Badge>
                {selectedNode.jurisdiction && (
                  <Badge variant="default">{selectedNode.jurisdiction}</Badge>
                )}
                {goodLawBadge(selectedNode)}
              </div>
              <dl className="mt-3 space-y-1 text-xs">
                {selectedNode.court && (
                  <div className="flex justify-between gap-2">
                    <dt className="text-muted">Court</dt>
                    <dd className="text-foreground text-right break-words">{selectedNode.court}</dd>
                  </div>
                )}
                {selectedNode.year != null && (
                  <div className="flex justify-between gap-2">
                    <dt className="text-muted">Year</dt>
                    <dd className="text-foreground">{selectedNode.year}</dd>
                  </div>
                )}
                {selectedNode.authority_score != null && (
                  <div className="flex justify-between gap-2">
                    <dt className="text-muted">Influence</dt>
                    <dd className="text-foreground">{selectedNode.authority_score.toFixed(3)}</dd>
                  </div>
                )}
              </dl>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
