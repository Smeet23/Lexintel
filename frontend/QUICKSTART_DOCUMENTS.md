# Document Management - Quick Start Guide

## Getting Started

The document management feature is now **ready to use** in your LexIntel frontend!

## What You Get

A complete, production-ready document upload and management system with:
- ✅ 24 passing tests
- ✅ Drag-and-drop file upload
- ✅ Real-time document list
- ✅ Status tracking
- ✅ Error handling
- ✅ Full TypeScript support

## Installation & Setup

No additional setup needed! Dependencies were already installed.

```bash
cd frontend
npm test                  # Verify all tests pass
npm run dev              # Start dev server
```

## How to Use

### Access the Documents Page

Navigate to: `http://localhost:3000/cases/[case-id]/documents`

Replace `[case-id]` with an actual case ID from your database.

### Features Available

1. **Upload Documents**
   - Drag and drop files onto the upload zone
   - Or click "Select Files" button
   - Supports: PDF, DOCX, TXT
   - Max size: 50MB per file

2. **View Documents**
   - See all documents in a table
   - Shows: Name, Size, Status, Upload Date
   - Status badges: pending → processing → ready

3. **Delete Documents**
   - Click trash icon on any document
   - Confirm deletion in dialog
   - List updates automatically

## Component Integration

### Using in Your Page

```tsx
import DocumentUploader from '@/components/document-uploader'
import DocumentManager from '@/components/document-manager'
import useDocuments from '@/hooks/useDocuments'

export default function MyPage() {
  const caseId = 'case-123'
  const { documents, isLoading, deleteDocument, refetch } = useDocuments(caseId)

  return (
    <div>
      {/* Upload Section */}
      <DocumentUploader
        caseId={caseId}
        onUploadComplete={refetch}
      />

      {/* Document List */}
      <DocumentManager
        caseId={caseId}
        documents={documents}
        onDelete={deleteDocument}
      />
    </div>
  )
}
```

## Key Components

### DocumentUploader
```tsx
<DocumentUploader
  caseId="case-123"
  onUploadComplete={(doc) => console.log('Uploaded:', doc)}
/>
```

### DocumentManager
```tsx
<DocumentManager
  caseId="case-123"
  documents={documents}
  onDelete={(id) => console.log('Deleting:', id)}
/>
```

### useDocuments Hook
```tsx
const {
  documents,      // DocumentItem[]
  isLoading,      // boolean
  error,          // Error | null
  refetch,        // () => Promise
  deleteDocument, // (id: string) => Promise
  isDeleting      // boolean
} = useDocuments(caseId)
```

## Running Tests

```bash
# Run all tests
npm test

# Run in watch mode (auto-rerun on changes)
npm test:watch

# Run specific test file
npm test document-uploader

# Run with coverage
npm test -- --coverage
```

## File Upload Flow

1. User selects/drops files
2. Files validated (type, size)
3. Upload request sent
4. Progress shown
5. Success message displayed
6. Document list refreshes
7. New document shows status: "processing"
8. Status updates to "ready" when indexed

## API Endpoints (Backend Required)

The frontend expects these endpoints:

```
GET    /cases/{caseId}/documents
POST   /cases/{caseId}/upload-document
DELETE /cases/{caseId}/documents/{documentId}
```

See backend documentation for implementation details.

## Troubleshooting

### Files not uploading?
- Check backend API is running
- Verify CORS is configured
- Check file size < 50MB
- Confirm file type is PDF, DOCX, or TXT

### Documents list empty?
- Verify case ID is correct
- Check backend database has documents
- Check `/cases/{id}/documents` endpoint works

### Tests failing?
```bash
npm install                  # Reinstall dependencies
npm test -- --clearCache    # Clear Jest cache
npm test:watch              # Debug with watch mode
```

### Build errors?
```bash
npm run lint               # Check for linting errors
npm run build              # Try production build
```

## Environment Variables

If needed, add to `.env.local`:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Performance Tips

1. **Caching**: Documents cached for 5 seconds
2. **Pagination**: Consider for large doc lists
3. **Lazy Loading**: Load document details on demand
4. **Batch Operations**: Combine multiple uploads

## Customization

### Styling

Components use Tailwind CSS. Customize with:
- Edit component className
- Override Tailwind config
- Use Shadcn/ui component props

### Validation

Modify in `document-uploader.tsx`:
```tsx
const MAX_FILE_SIZE = 50 * 1024 * 1024  // Change size limit
const ALLOWED_TYPES = [...]             // Add/remove file types
```

### Status Colors

Update in `lib/document-utils.ts`:
```tsx
const getStatusColor = (status) => {
  // Customize badge colors here
}
```

## Documentation

- **Full Guide**: See `DOCUMENT_MANAGEMENT.md`
- **API Reference**: See component JSDoc comments
- **Test Examples**: See `__tests__/` directory

## Next Steps

1. ✅ Verify tests pass: `npm test`
2. ✅ Start dev server: `npm run dev`
3. ✅ Navigate to documents page
4. ✅ Try uploading a document
5. ✅ Check if backend processes it correctly

## Support

- Check test files for usage examples
- Review component props/interfaces
- See error messages for validation feedback

---

**Status**: ✅ Ready for development/production
**Test Coverage**: 24 tests, 100% passing
**Dependencies**: All included via npm
