import type { Metadata } from 'next'
import './globals.css'
import { AuthProvider } from '@/lib/auth-context'
import { NavBar } from '@/components/navbar'

export const metadata: Metadata = {
  title: 'LexIntel - Legal RAG',
  description: 'RAG system for legal document analysis',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <AuthProvider>
      <html lang="en">
        <body className="bg-gray-50">
          <NavBar />
          <main className="max-w-7xl mx-auto px-4 py-8">
            {children}
          </main>
        </body>
      </html>
    </AuthProvider>
  )
}
