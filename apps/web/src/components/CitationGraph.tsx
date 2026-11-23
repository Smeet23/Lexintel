"use client"

import React, { useCallback, useEffect, useMemo } from "react"
import {
  ReactFlow,
  ReactFlowProvider,
  Controls,
  MiniMap,
  Background,
  useNodesState,
  useEdgesState,
  MarkerType,
  Position,
  Handle,
  type Node,
  type Edge,
  type NodeProps,
} from "@xyflow/react"
import "@xyflow/react/dist/style.css"
import dagre from "@dagrejs/dagre"
import {
  Scale,
  FileText,
  BookOpen,
  CheckCircle2,
  XCircle,
  HelpCircle,
} from "lucide-react"
import type { GraphNode, GraphEdge, TreatmentType } from "@/lib/types"

// NOTE: ReactFlow must be imported from the SAME module instance as Handle/
// ReactFlowProvider/hooks above. A previous `require("@xyflow/react").ReactFlow`
// hack pulled in a SECOND module instance, so <Handle> could not find the React
// Flow context — producing hundreds of "Handle: No node id found" warnings.

// ============================================
// Internal data type for node
// ============================================

interface CaseNodeData extends Record<string, unknown> {
  graphNode: GraphNode
  isCenter: boolean
}

// ============================================
// Status + Icon helpers
// ============================================

// Keyed on the backend's honest good_law_status. "no_adverse" (a real authority
// with no overruling found in this graph) is a genuine positive signal — shown
// green but labelled distinctly from a CourtListener-confirmed "Good law", and
// never as the misleading "Not computed".
function getStatusColors(status: string | undefined) {
  switch (status) {
    case "good":
      return {
        label: "Good law",
        bg: "bg-green-50", border: "border-green-400",
        badge: "bg-green-100 text-green-800",
        icon: <CheckCircle2 className="w-3 h-3 text-green-500" />,
      }
    case "no_adverse":
      return {
        label: "No adverse treatment",
        bg: "bg-emerald-50", border: "border-emerald-300",
        badge: "bg-emerald-100 text-emerald-800",
        icon: <CheckCircle2 className="w-3 h-3 text-emerald-500" />,
      }
    case "bad":
      return {
        label: "Bad law",
        bg: "bg-red-50", border: "border-red-400",
        badge: "bg-red-100 text-red-800",
        icon: <XCircle className="w-3 h-3 text-red-500" />,
      }
    default:  // "unknown" / undefined (e.g. document provenance nodes)
      return {
        label: "Not validated",
        bg: "bg-gray-50", border: "border-gray-300",
        badge: "bg-gray-100 text-gray-700",
        icon: <HelpCircle className="w-3 h-3 text-gray-400" />,
      }
  }
}

// Resolve a node's status, falling back to the legacy boolean is_good_law when
// good_law_status is absent (older payloads).
function nodeGoodLawStatus(node: GraphNode): string {
  if (node.good_law_status) return node.good_law_status
  if (node.is_good_law === false) return "bad"
  if (node.is_good_law === true) return "good"
  return "unknown"
}

function getTypeIcon(type: GraphNode["type"]) {
  switch (type) {
    case "case":
      return <Scale className="w-3.5 h-3.5" />
    case "statute":
      return <BookOpen className="w-3.5 h-3.5" />
    case "regulation":
      return <FileText className="w-3.5 h-3.5" />
    default:
      return <Scale className="w-3.5 h-3.5" />
  }
}

// ============================================
// Treatment edge styles
// ============================================

function getTreatmentEdgeStyle(treatment: TreatmentType) {
  switch (treatment) {
    case "FOLLOWS":
    case "AFFIRMS":
    case "APPLIES":
      return {
        stroke: "#16a34a",
        strokeWidth: 2,
        strokeDasharray: undefined as string | undefined,
        animated: false,
      }
    case "OVERRULES":
    case "REVERSES":
      return {
        stroke: "#dc2626",
        strokeWidth: 2.5,
        strokeDasharray: "6 3",
        animated: true,
      }
    case "ABROGATES":
    case "SUPERSEDES":
      return {
        stroke: "#b91c1c",
        strokeWidth: 2,
        strokeDasharray: "4 2",
        animated: true,
      }
    case "QUESTIONS":
    case "CRITICIZES":
      return {
        stroke: "#d97706",
        strokeWidth: 1.5,
        strokeDasharray: "2 2",
        animated: false,
      }
    case "DISTINGUISHES":
      return {
        stroke: "#7c3aed",
        strokeWidth: 1.5,
        strokeDasharray: undefined,
        animated: false,
      }
    case "AMENDS":
    case "REPEALS":
    case "CODIFIES":
    case "INTERPRETS":
      return {
        stroke: "#2563eb",
        strokeWidth: 1.5,
        strokeDasharray: "5 2",
        animated: false,
      }
    case "CITES":
    case "LIMITS":
    case "EXPLAINS":
    case "HARMONIZES":
    default:
      return {
        stroke: "#6b7280",
        strokeWidth: 1,
        strokeDasharray: undefined,
        animated: false,
      }
  }
}

// ============================================
// Custom CaseNode component
// ============================================

function CaseNode({ data }: NodeProps) {
  const nodeData = data as CaseNodeData
  const { graphNode, isCenter } = nodeData
  const colors = getStatusColors(nodeGoodLawStatus(graphNode))
  const typeIcon = getTypeIcon(graphNode.type)
  const displayName = graphNode.name ?? graphNode.citation

  return (
    <div
      className={[
        "rounded-lg border-2 px-3 py-2 shadow-sm cursor-pointer transition-all",
        "min-w-[180px] max-w-[220px]",
        colors.bg,
        colors.border,
        isCenter ? "ring-2 ring-offset-2 ring-blue-500" : "",
      ].join(" ")}
    >
      {/* React Flow needs source/target Handles on a custom node to anchor
          edges — without them edges silently don't render (nodes still show).
          Hidden because connections are programmatic, not user-drawn. */}
      <Handle type="target" position={Position.Top} style={{ opacity: 0 }} isConnectable={false} />
      <Handle type="source" position={Position.Bottom} style={{ opacity: 0 }} isConnectable={false} />

      {/* Header row */}
      <div className="flex items-center gap-1.5 mb-1">
        <span className="text-gray-500">{typeIcon}</span>
        <span
          className={`text-xs px-1.5 py-0.5 rounded font-medium flex items-center gap-1 ${colors.badge}`}
        >
          {colors.icon}
          <span>{colors.label}</span>
        </span>
      </div>

      {/* Case name */}
      <div
        className="text-xs font-semibold text-gray-800 leading-tight truncate"
        title={displayName}
      >
        {displayName}
      </div>

      {/* Court + year */}
      <div className="flex items-center gap-1 mt-1 flex-wrap">
        {graphNode.court && (
          <span
            className="text-[10px] text-gray-500 truncate max-w-[100px]"
            title={graphNode.court}
          >
            {graphNode.court}
          </span>
        )}
        {graphNode.year && (
          <span className="text-[10px] text-gray-400">({graphNode.year})</span>
        )}
      </div>

      {/* Jurisdiction + authority score */}
      {(graphNode.jurisdiction || graphNode.authority_score !== undefined) && (
        <div className="flex items-center justify-between mt-1">
          {graphNode.jurisdiction && (
            <span className="text-[10px] bg-blue-50 text-blue-700 px-1 rounded">
              {graphNode.jurisdiction}
            </span>
          )}
          {graphNode.authority_score !== undefined && (
            <span className="text-[10px] text-gray-400">
              {(graphNode.authority_score * 100).toFixed(0)}%
            </span>
          )}
        </div>
      )}
    </div>
  )
}

const nodeTypes = { caseNode: CaseNode }

// ============================================
// Dagre layout
// ============================================

const NODE_WIDTH = 200
const NODE_HEIGHT = 80

function applyDagreLayout(
  rfNodes: Node[],
  rfEdges: Edge[]
): { nodes: Node[]; edges: Edge[] } {
  const g = new dagre.graphlib.Graph()
  g.setDefaultEdgeLabel(() => ({}))
  g.setGraph({ rankdir: "TB", nodesep: 40, ranksep: 60 })

  for (const node of rfNodes) {
    g.setNode(node.id, { width: NODE_WIDTH, height: NODE_HEIGHT })
  }
  for (const edge of rfEdges) {
    g.setEdge(edge.source, edge.target)
  }

  dagre.layout(g)

  const laidOutNodes = rfNodes.map((node) => {
    const dagreNode = g.node(node.id)
    return {
      ...node,
      position: {
        x: dagreNode.x - NODE_WIDTH / 2,
        y: dagreNode.y - NODE_HEIGHT / 2,
      },
      sourcePosition: Position.Bottom,
      targetPosition: Position.Top,
    }
  })

  return { nodes: laidOutNodes, edges: rfEdges }
}

// ============================================
// Build React Flow elements from graph data
// ============================================

function buildReactFlowElements(
  graphNodes: GraphNode[],
  graphEdges: GraphEdge[],
  centerCitation: string | undefined,
): { nodes: Node[]; edges: Edge[] } {
  const rfNodes: Node[] = graphNodes.map((gn) => ({
    id: gn.id,
    type: "caseNode",
    position: { x: 0, y: 0 },
    data: {
      graphNode: gn,
      isCenter: gn.citation === centerCitation,
    } as CaseNodeData,
  }))

  const rfEdges: Edge[] = graphEdges.map((ge, idx) => {
    const style = getTreatmentEdgeStyle(ge.treatment)
    return {
      id: `edge-${idx}-${ge.source}-${ge.target}`,
      source: ge.source,
      target: ge.target,
      animated: style.animated,
      label: ge.treatment,
      labelStyle: { fontSize: 9, fill: style.stroke },
      style: {
        stroke: style.stroke,
        strokeWidth: style.strokeWidth,
        strokeDasharray: style.strokeDasharray,
      },
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: style.stroke,
        width: 14,
        height: 14,
      },
    }
  })

  return applyDagreLayout(rfNodes, rfEdges)
}

// ============================================
// Inner flow component (must be inside ReactFlowProvider)
// ============================================

interface InnerFlowProps {
  graphNodes: GraphNode[]
  graphEdges: GraphEdge[]
  centerCitation?: string
  onNodeClick?: (node: GraphNode) => void
}

function InnerFlow({ graphNodes, graphEdges, centerCitation, onNodeClick }: InnerFlowProps) {
  const { nodes: initialNodes, edges: initialEdges } = useMemo(
    () => buildReactFlowElements(graphNodes, graphEdges, centerCitation),
    [graphNodes, graphEdges, centerCitation]
  )

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges)

  // useNodesState/useEdgesState only seed from their argument on FIRST mount.
  // The graph data loads asynchronously (React Query), so the first render has
  // empty nodes/edges and those empties get captured. When the real data
  // arrives the memo recomputes but the RF state is never updated — edges (and
  // re-laid-out nodes) silently vanish. Re-sync whenever the computed elements
  // change so async-loaded edges actually render.
  useEffect(() => {
    setNodes(initialNodes)
    setEdges(initialEdges)
  }, [initialNodes, initialEdges, setNodes, setEdges])

  const handleNodeClick = useCallback(
    (_event: React.MouseEvent, node: Node) => {
      const nodeData = node.data as CaseNodeData
      onNodeClick?.(nodeData.graphNode)
    },
    [onNodeClick]
  )

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onNodeClick={handleNodeClick}
      nodeTypes={nodeTypes}
      fitView
      minZoom={0.3}
      maxZoom={2}
      nodesDraggable
      nodesConnectable={false}
      attributionPosition="bottom-right"
    >
      <Controls />
      <MiniMap
        nodeColor={(node: Node) => {
          const d = node.data as CaseNodeData
          switch (nodeGoodLawStatus(d?.graphNode as GraphNode)) {
            case "good":
              return "#16a34a"
            case "no_adverse":
              return "#10b981"
            case "bad":
              return "#dc2626"
            default:
              return "#9ca3af"
          }
        }}
        maskColor="rgba(255,255,255,0.7)"
      />
      <Background gap={16} color="#e5e7eb" />
    </ReactFlow>
  )
}

// ============================================
// Public CitationGraph component
// ============================================

interface CitationGraphProps {
  graphNodes: GraphNode[]
  graphEdges: GraphEdge[]
  centerCitation?: string
  className?: string
  onNodeClick?: (node: GraphNode) => void
}

export default function CitationGraph({
  graphNodes,
  graphEdges,
  centerCitation,
  className = "",
  onNodeClick,
}: CitationGraphProps) {
  if (!graphNodes.length) {
    return (
      <div
        className={`flex items-center justify-center text-gray-400 text-sm ${className}`}
        style={{ minHeight: 200 }}
      >
        No citation graph data available.
      </div>
    )
  }

  return (
    // ReactFlow needs an explicit (non-percentage-resolving) height on its
    // container, otherwise .react-flow collapses to height:0 and nodes are
    // laid out but invisible. minHeight alone does not satisfy height:100%.
    <div className={`w-full ${className}`} style={{ height: 480 }}>
      <ReactFlowProvider>
        <InnerFlow
          graphNodes={graphNodes}
          graphEdges={graphEdges}
          centerCitation={centerCitation}
          onNodeClick={onNodeClick}
        />
      </ReactFlowProvider>
    </div>
  )
}
