/**
 * Document utility functions for formatting and status management
 */

export const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i]
}

export const formatDate = (dateString: string): string => {
  const date = new Date(dateString)
  return date.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  })
}

export const getStatusColor = (
  status: 'pending' | 'processing' | 'ready' | 'error'
): 'default' | 'secondary' | 'destructive' => {
  switch (status) {
    case 'ready':
      return 'default'
    case 'processing':
    case 'pending':
      return 'secondary'
    case 'error':
      return 'destructive'
    default:
      return 'default'
  }
}
