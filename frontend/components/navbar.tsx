'use client';

import Link from 'next/link'
import { useAuth } from '@/lib/auth-context'

export function NavBar() {
  const { isAuthenticated, setToken } = useAuth();

  const handleLogout = () => {
    setToken(null);
    window.location.href = '/';
  };

  return (
    <nav className="bg-white shadow">
      <div className="max-w-7xl mx-auto px-4 py-4 flex justify-between items-center">
        <Link href="/" className="text-2xl font-bold text-blue-600 hover:text-blue-700">
          LexIntel
        </Link>
        <div className="flex gap-4 items-center">
          {isAuthenticated && (
            <>
              <Link href="/dashboard" className="text-gray-700 hover:text-gray-900">
                Dashboard
              </Link>
              <button
                onClick={handleLogout}
                className="bg-red-600 text-white px-4 py-2 rounded hover:bg-red-700"
              >
                Logout
              </button>
            </>
          )}
        </div>
      </div>
    </nav>
  );
}
