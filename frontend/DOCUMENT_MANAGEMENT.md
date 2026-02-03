# Document Management Feature

This guide explains the document upload and management system for LexIntel.

## Overview

The document management feature allows users to:
- Upload legal documents (PDF, DOCX, TXT)
- View uploaded documents with processing status
- Delete documents
- Monitor real-time document processing status

## Architecture

### Components

#### DocumentUploader (`components/document-uploader.tsx`)
Handles file upload functionality with drag-and-drop support.

**Props:**
- `caseId: string` - The case ID to upload documents to
- `onUploadComplete?: (document: UploadedDocument) => void` - Callback after successful upload

**Features:**
- Drag and drop file selection
- Click to browse file picker
- File type validation (PDF, DOCX, TXT)
- File size validation (max 50MB)
- Real-time upload progress
- Error and success messaging

**Example:**
```tsx
<DocumentUploader
  caseId="case-123"
  onUploadComplete={() => refetchDocuments()}
/>
```

#### DocumentManager (`components/document-manager.tsx`)
Displays a list of uploaded documents with metadata and actions.

**Props:**
- `caseId: string` - The case ID
- `documents: DocumentItem[]` - Array of document objects
- `onDelete?: (documentId: string) => void` - Callback for document deletion

**Features:**
- Document list table with sorting capability
- Status badges (pending, processing, ready, error)
- File size formatting (B, KB, MB, GB)
- Upload date formatting
- Delete action with confirmation dialog

**Example:**
```tsx
<DocumentManager
  caseId="case-123"
  documents={documents}
  onDelete={handleDelete}
/>
```

### Hooks

#### useDocuments (`hooks/useDocuments.ts`)
Custom React Query hook for managing document data and operations.

**Returns:**
```typescript
{
  documents: DocumentItem[]        // Array of documents
  isLoading: boolean               // Loading state
  error: Error | null              // Error object if failed
  refetch: () => void              // Refetch documents
  deleteDocument: (id: string) => Promise<void>  // Delete function
  isDeleting: boolean              // Delete loading state
}
```

**Usage:**
```tsx
const { documents, isLoading, error, deleteDocument } = useDocuments('case-123')
```

### Pages

#### Documents Page (`app/cases/[id]/documents/page.tsx`)
The main page for document management within a case.

**Features:**
- Displays upload interface
- Shows document list
- Handles error states
- Loading indicators
- Integrates all components

## Data Models

### DocumentItem
```typescript
interface DocumentItem {
  id: string                                              // Unique document ID
  filename: string                                        // Document name
  size: number                                            // File size in bytes
  status: 'pending' | 'processing' | 'ready' | 'error'  // Document status
  created_at: string                                      // ISO date string
}
```

## Utility Functions

### formatFileSize (`lib/document-utils.ts`)
Converts bytes to human-readable format.

```typescript
formatFileSize(1048576) // Returns "1 MB"
```

### formatDate (`lib/document-utils.ts`)
Formats ISO date strings to readable format.

```typescript
formatDate("2024-01-15T14:30:00Z") // Returns "Jan 15, 2024"
```

### getStatusColor (`lib/document-utils.ts`)
Returns badge color variant based on status.

```typescript
getStatusColor('ready')       // Returns 'default'
getStatusColor('processing')  // Returns 'secondary'
getStatusColor('error')       // Returns 'destructive'
```

## API Integration

### Endpoints Used

**GET `/cases/{caseId}/documents`**
- Fetches list of documents for a case
- Returns: `{ documents: DocumentItem[] }`

**POST `/cases/{caseId}/upload-document`**
- Upload a document
- Form data: `file` (multipart), `case_id`
- Returns: `{ id, filename, status }`

**DELETE `/cases/{caseId}/documents/{documentId}`**
- Delete a document
- Returns: `{ success: true }`

## Testing

### Test Files
- `__tests__/components/document-uploader.test.tsx` - Upload component tests
- `__tests__/components/document-manager.test.tsx` - Document list tests
- `__tests__/hooks/useDocuments.test.tsx` - Hook tests
- `__tests__/pages/documents-page.test.tsx` - Page integration tests

### Running Tests
```bash
npm test                    # Run all tests
npm test:watch             # Run in watch mode
npm test -- --coverage     # Run with coverage
```

## File Validation

**Allowed Types:**
- application/pdf (`.pdf`)
- application/vnd.openxmlformats-officedocument.wordprocessingml.document (`.docx`)
- text/plain (`.txt`)

**Size Limits:**
- Maximum 50MB per file
- Validated on client and server

## Error Handling

The system handles errors gracefully:

1. **Invalid File Type**: User is notified to upload correct format
2. **File Too Large**: User is told maximum file size is 50MB
3. **Upload Failure**: Error message displayed with retry option
4. **Fetch Failure**: Alert shown when unable to load documents
5. **Delete Failure**: Error message shown when unable to delete

## User Experience

### Upload Flow
1. User drags file onto drop zone OR clicks "Select Files"
2. File is validated (type and size)
3. Upload progress is shown
4. Success message appears
5. Document list automatically refreshes
6. New document shows with "processing" status

### Status Progression
- `pending`: Document queued for processing
- `processing`: Document is being indexed
- `ready`: Document is indexed and queryable
- `error`: Processing failed, document unusable

### Deletion Flow
1. User clicks delete icon on document row
2. Confirmation dialog appears
3. User confirms deletion
4. Document is deleted
5. List refreshes automatically

## Performance

- **Queries**: Cached for 5 seconds to reduce API calls
- **Mutations**: Real-time invalidation on delete
- **Rendering**: Virtual table for large document lists (future)
- **Upload**: Progressive upload with real-time feedback

## Accessibility

- Proper ARIA labels for buttons
- Keyboard navigation support
- Status updates announced to screen readers
- Form validation messages provided to users

## Future Enhancements

- Document preview/download
- Bulk upload operations
- Document renaming and descriptions
- Advanced filtering and sorting
- Document versioning
- Export document list
- Document analytics
