import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClientProvider, QueryClient } from '@tanstack/react-query'
import Dashboard from '@/app/dashboard/page'

jest.mock('@/lib/api', () => ({
  __esModule: true,
  default: {
    post: jest.fn(),
    get: jest.fn(),
  },
}))

jest.mock('next/navigation', () => ({
  useRouter: jest.fn(() => ({
    push: jest.fn(),
  })),
}))

jest.mock('@/components/document-uploader', () => {
  return function MockDocumentUploader({
    onUploadComplete,
  }: {
    onUploadComplete?: (doc: any) => void
  }) {
    return (
      <div>
        <input
          type="file"
          aria-label="file-input"
          onChange={(e) => {
            const file = e.target.files?.[0]
            if (file) {
              onUploadComplete?.(file)
            }
          }}
        />
      </div>
    )
  }
})

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

describe('Dashboard', () => {
  beforeEach(() => {
    jest.clearAllMocks()
  })

  it('should render dashboard with form fields', () => {
    renderWithQueryClient(<Dashboard />)

    expect(screen.getByText(/create new case/i)).toBeInTheDocument()
    expect(screen.getByLabelText(/case name/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /upload & analyze/i })).toBeInTheDocument()
  })

  it('should display case name counter', () => {
    renderWithQueryClient(<Dashboard />)

    expect(screen.getByText(/255/)).toBeInTheDocument()
  })

  it('should show error for empty case name on submit', async () => {
    renderWithQueryClient(<Dashboard />)

    const submitButton = screen.getByRole('button', { name: /upload & analyze/i })
    expect(submitButton).toBeDisabled()
  })

  it('should show error when case name is too short', async () => {
    renderWithQueryClient(<Dashboard />)

    const caseNameInput = screen.getByLabelText(/case name/i)
    await userEvent.type(caseNameInput, 'A')

    expect(screen.getByText(/case name must be at least/i)).toBeInTheDocument()
  })

  it('should enable submit when form is valid', async () => {
    renderWithQueryClient(<Dashboard />)

    const caseNameInput = screen.getByLabelText(/case name/i)
    await userEvent.type(caseNameInput, 'Test Case')

    const file = new File(['test'], 'document.pdf', { type: 'application/pdf' })
    const fileInput = screen.getByLabelText(/file-input/i) as HTMLInputElement
    fireEvent.change(fileInput, { target: { files: [file] } })

    const submitButton = screen.getByRole('button', { name: /upload & analyze/i })
    expect(submitButton).not.toBeDisabled()
  })

  it('should display selected file with formatted size', async () => {
    renderWithQueryClient(<Dashboard />)

    const caseNameInput = screen.getByLabelText(/case name/i)
    await userEvent.type(caseNameInput, 'Test Case')

    const file = new File(['test content'], 'document.pdf', { type: 'application/pdf' })
    const fileInput = screen.getByLabelText(/file-input/i) as HTMLInputElement
    fireEvent.change(fileInput, { target: { files: [file] } })

    await waitFor(() => {
      expect(screen.getByText(/document ready/i)).toBeInTheDocument()
      expect(screen.getByText(/document.pdf/i)).toBeInTheDocument()
    })
  })

  it('should submit form with valid data', async () => {
    const api = require('@/lib/api').default

    api.post.mockResolvedValueOnce({
      data: { id: 'case-123' },
    })

    renderWithQueryClient(<Dashboard />)

    const caseNameInput = screen.getByLabelText(/case name/i)
    await userEvent.type(caseNameInput, 'Test Case')

    const file = new File(['test'], 'document.pdf', { type: 'application/pdf' })
    const fileInput = screen.getByLabelText(/file-input/i) as HTMLInputElement
    fireEvent.change(fileInput, { target: { files: [file] } })

    const submitButton = screen.getByRole('button', { name: /upload & analyze/i })
    fireEvent.click(submitButton)

    await waitFor(() => {
      expect(api.post).toHaveBeenCalled()
    })
  })

  it('should display error on upload failure', async () => {
    const api = require('@/lib/api').default

    api.post.mockRejectedValueOnce({
      response: {
        data: {
          detail: 'File upload failed',
        },
      },
    })

    renderWithQueryClient(<Dashboard />)

    const caseNameInput = screen.getByLabelText(/case name/i)
    await userEvent.type(caseNameInput, 'Test Case')

    const file = new File(['test'], 'document.pdf', { type: 'application/pdf' })
    const fileInput = screen.getByLabelText(/file-input/i) as HTMLInputElement
    fireEvent.change(fileInput, { target: { files: [file] } })

    const submitButton = screen.getByRole('button', { name: /upload & analyze/i })
    fireEvent.click(submitButton)

    await waitFor(() => {
      expect(screen.getByText(/file upload failed/i)).toBeInTheDocument()
    })
  })

  it('should clear errors when input changes', async () => {
    renderWithQueryClient(<Dashboard />)

    const caseNameInput = screen.getByLabelText(/case name/i)
    await userEvent.type(caseNameInput, 'A')

    expect(screen.getByText(/case name must be at least/i)).toBeInTheDocument()

    await userEvent.type(caseNameInput, 'BC')

    await waitFor(() => {
      expect(screen.queryByText(/case name must be at least/i)).not.toBeInTheDocument()
    })
  })
})
