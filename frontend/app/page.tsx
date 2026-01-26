'use client'

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link'
import { useAuth } from '@/lib/auth-context';

export default function Home() {
  const router = useRouter();
  const { isAuthenticated } = useAuth();

  useEffect(() => {
    if (isAuthenticated) {
      router.push('/dashboard');
    }
  }, [isAuthenticated, router]);

  if (isAuthenticated) {
    return null;
  }

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
