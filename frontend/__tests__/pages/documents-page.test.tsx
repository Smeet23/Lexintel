import React from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import { QueryClientProvider, QueryClient } from '@tanstack/react-query'
import DocumentsPage from '@/app/cases/[id]/documents/page'

jest.mock('@/lib/api', () => ({
  __esModule: true,
  default: {
    get: jest.fn(),
    post: jest.fn(),
    delete: jest.fn(),
  },
}))

jest.mock('next/navigation', () => ({
  useParams: jest.fn(() => ({ id: 'case-123' })),
}))

const createTestQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: { retry: false, gcTime: 0 },
      mutations: { retry: false },
    },
  })

const renderWithQueryClient = (component: React.ReactElement) => {
  const testQueryClient = createTestQueryClient()
  return render(
    <QueryClientProvider client={testQueryClient}>
      {component}
    </QueryClientProvider>
  )
}

describe('DocumentsPage', () => {
  beforeEach(() => {
    jest.clearAllMocks()
  })

  it('should render documents page with upload and list sections', async () => {
    const api = require('@/lib/api').default

    api.get.mockResolvedValueOnce({
      data: {
        documents: [],
      },
    })

    renderWithQueryClient(<DocumentsPage />)

    expect(screen.getByText(/upload documents/i)).toBeInTheDocument()
  })

  it('should fetch and display list of documents', async () => {
    const api = require('@/lib/api').default

    api.get.mockResolvedValueOnce({
      data: {
        documents: [
          {
            id: '1',
            filename: 'contract.pdf',
            size: 1024,
            status: 'ready',
            created_at: '2024-01-01T00:00:00Z',
          },
        ],
      },
    })

    renderWithQueryClient(<DocumentsPage />)

    await waitFor(
      () => {
        expect(screen.getByText('contract.pdf')).toBeInTheDocument()
      },
      { timeout: 3000 }
    )
  })

  it('should refresh document list on mount', async () => {
    const api = require('@/lib/api').default

    api.get.mockResolvedValueOnce({
      data: {
        documents: [],
      },
    })

    renderWithQueryClient(<DocumentsPage />)

    await waitFor(() => {
      expect(api.get).toHaveBeenCalled()
    })
  })

  it('should handle API errors gracefully', async () => {
    const api = require('@/lib/api').default

    api.get.mockRejectedValueOnce(new Error('Failed to fetch documents'))

    renderWithQueryClient(<DocumentsPage />)

    await waitFor(
      () => {
        expect(screen.getByText(/failed to load documents/i)).toBeInTheDocument()
      },
      { timeout: 3000 }
    )
  })
})
