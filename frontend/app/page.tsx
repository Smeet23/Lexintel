'use client'

import Link from 'next/link'

export default function Home() {
  return (
    <div className="text-center py-12">
      <h2 className="text-4xl font-bold mb-4">Welcome to LexIntel</h2>
      <p className="text-xl text-gray-600 mb-8">Legal Document Analysis with RAG</p>
      <div className="space-x-4">
        <Link href="/auth/login" className="bg-blue-600 text-white px-6 py-3 rounded hover:bg-blue-700">
          Login
        </Link>
        <Link href="/auth/register" className="bg-gray-600 text-white px-6 py-3 rounded hover:bg-gray-700">
          Register
        </Link>
      </div>
    </div>
  )
}
