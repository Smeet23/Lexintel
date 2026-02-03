import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { QueryClientProvider, QueryClient } from '@tanstack/react-query'
import DocumentManager from '@/components/document-manager'

// Mock axios
jest.mock('@/lib/api', () => ({
  __esModule: true,
  default: {
    get: jest.fn(),
    post: jest.fn(),
  },
}))

const createTestQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: { retry: false },
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

describe('DocumentManager', () => {
  beforeEach(() => {
    jest.clearAllMocks()
  })

  it('should render document list with status badges', async () => {
    const mockDocuments = [
      {
        id: '1',
        filename: 'contract.pdf',
        size: 1024,
        status: 'ready',
        created_at: '2024-01-01T00:00:00Z',
      },
      {
        id: '2',
        filename: 'agreement.docx',
        size: 2048,
        status: 'processing',
        created_at: '2024-01-02T00:00:00Z',
      },
    ]

    renderWithQueryClient(<DocumentManager caseId="case-123" documents={mockDocuments} />)

    await waitFor(() => {
      expect(screen.getByText('contract.pdf')).toBeInTheDocument()
      expect(screen.getByText('agreement.docx')).toBeInTheDocument()
    })

    expect(screen.getByText('ready')).toBeInTheDocument()
    expect(screen.getByText('processing')).toBeInTheDocument()
  })

  it('should display file size in human-readable format', async () => {
    const mockDocuments = [
      {
        id: '1',
        filename: 'large-document.pdf',
        size: 1048576, // 1MB
        status: 'ready',
        created_at: '2024-01-01T00:00:00Z',
      },
    ]

    renderWithQueryClient(<DocumentManager caseId="case-123" documents={mockDocuments} />)

    await waitFor(() => {
      expect(screen.getByText('1 MB')).toBeInTheDocument()
    })
  })

  it('should call delete handler when delete button clicked', async () => {
    const mockDocuments = [
      {
        id: '1',
        filename: 'contract.pdf',
        size: 1024,
        status: 'ready',
        created_at: '2024-01-01T00:00:00Z',
      },
    ]
    const onDeleteMock = jest.fn()

    renderWithQueryClient(
      <DocumentManager
        caseId="case-123"
        documents={mockDocuments}
        onDelete={onDeleteMock}
      />
    )

    const deleteButton = screen.getAllByRole('button')[0]
    fireEvent.click(deleteButton)

    await waitFor(() => {
      expect(screen.getByText(/confirm/i)).toBeInTheDocument()
    })

    const confirmDeleteButton = screen.getByRole('button', { name: /delete/i })
    fireEvent.click(confirmDeleteButton)

    await waitFor(() => {
      expect(onDeleteMock).toHaveBeenCalledWith('1')
    })
  })

  it('should show confirmation dialog before deleting document', async () => {
    const mockDocuments = [
      {
        id: '1',
        filename: 'contract.pdf',
        size: 1024,
        status: 'ready',
        created_at: '2024-01-01T00:00:00Z',
      },
    ]

    renderWithQueryClient(<DocumentManager caseId="case-123" documents={mockDocuments} />)

    const deleteButton = screen.getAllByRole('button')[0]
    fireEvent.click(deleteButton)

    await waitFor(() => {
      expect(screen.getByText(/confirm/i)).toBeInTheDocument()
    })
  })

  it('should display empty state when no documents exist', () => {
    renderWithQueryClient(<DocumentManager caseId="case-123" documents={[]} />)

    expect(screen.getByText(/no documents/i)).toBeInTheDocument()
  })

  it('should update document status in real-time', async () => {
    const mockDocuments = [
      {
        id: '1',
        filename: 'contract.pdf',
        size: 1024,
        status: 'processing',
        created_at: '2024-01-01T00:00:00Z',
      },
    ]

    const { rerender } = renderWithQueryClient(
      <DocumentManager caseId="case-123" documents={mockDocuments} />
    )

    expect(screen.getByText('processing')).toBeInTheDocument()

    const updatedDocuments = [
      {
        ...mockDocuments[0],
        status: 'ready',
      },
    ]

    rerender(
      <QueryClientProvider client={new QueryClient()}>
        <DocumentManager caseId="case-123" documents={updatedDocuments} />
      </QueryClientProvider>
    )

    await waitFor(() => {
      expect(screen.getByText('ready')).toBeInTheDocument()
    })
  })

  it('should display upload date in readable format', async () => {
    const mockDocuments = [
      {
        id: '1',
        filename: 'contract.pdf',
        size: 1024,
        status: 'ready',
        created_at: '2024-01-15T14:30:00Z',
      },
    ]

    renderWithQueryClient(<DocumentManager caseId="case-123" documents={mockDocuments} />)

    await waitFor(() => {
      expect(screen.getByText(/jan 15/i)).toBeInTheDocument()
    })
  })
})
