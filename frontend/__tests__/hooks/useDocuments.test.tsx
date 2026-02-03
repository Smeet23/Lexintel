import { renderHook, waitFor } from '@testing-library/react'
import { QueryClientProvider, QueryClient } from '@tanstack/react-query'
import useDocuments from '@/hooks/useDocuments'
import api from '@/lib/api'

jest.mock('@/lib/api')

const createTestQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  })

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <QueryClientProvider client={createTestQueryClient()}>
    {children}
  </QueryClientProvider>
)

describe('useDocuments hook', () => {
  beforeEach(() => {
    jest.clearAllMocks()
  })

  it('should fetch documents for a case', async () => {
    const mockDocuments = [
      {
        id: '1',
        filename: 'contract.pdf',
        size: 1024,
        status: 'ready',
        created_at: '2024-01-01T00:00:00Z',
      },
    ]

    ;(api.get as jest.Mock).mockResolvedValueOnce({
      data: { documents: mockDocuments },
    })

    const { result } = renderHook(() => useDocuments('case-123'), { wrapper })

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false)
    })

    expect(result.current.documents).toEqual(mockDocuments)
  })

  it('should handle delete mutation', async () => {
    ;(api.get as jest.Mock).mockResolvedValueOnce({
      data: { documents: [] },
    })

    ;(api.delete as jest.Mock).mockResolvedValueOnce({ data: { success: true } })

    const { result } = renderHook(() => useDocuments('case-123'), { wrapper })

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false)
    })

    result.current.deleteDocument('doc-1')

    await waitFor(() => {
      expect(api.delete).toHaveBeenCalledWith('/cases/case-123/documents/doc-1')
    })
  })

  it('should handle errors', async () => {
    ;(api.get as jest.Mock).mockRejectedValueOnce(new Error('Network error'))

    const { result } = renderHook(() => useDocuments('case-123'), { wrapper })

    await waitFor(() => {
      expect(result.current.error).toBeDefined()
    })
  })
})
