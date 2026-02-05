'use client';

import React, { createContext, useContext, useState, useEffect } from 'react';

interface AuthContextType {
  token: string | null;
  isAuthenticated: boolean;
  setToken: (token: string | null) => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

// TODO: Remove after testing - auto-login with demo user
const AUTO_LOGIN_DEMO = true;
const DEMO_EMAIL = 'demo@example.com';
const DEMO_PASSWORD = 'Demo1234';

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [token, setTokenState] = useState<string | null>(null);
  const [isMounted, setIsMounted] = useState(false);

  // Load token from localStorage on mount
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const storedToken = localStorage.getItem('access_token');
      if (storedToken) {
        setTokenState(storedToken);
        setIsMounted(true);
      } else if (AUTO_LOGIN_DEMO) {
        // Auto-login with demo user for testing
        const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
        fetch(`${API_URL}/auth/login`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ email: DEMO_EMAIL, password: DEMO_PASSWORD }),
        })
          .then((res) => res.json())
          .then((data) => {
            if (data.access_token) {
              localStorage.setItem('access_token', data.access_token);
              setTokenState(data.access_token);
            }
            setIsMounted(true);
          })
          .catch(() => setIsMounted(true));
      } else {
        setIsMounted(true);
      }
    }
  }, []);

  const setToken = (newToken: string | null) => {
    setTokenState(newToken);
    if (typeof window !== 'undefined') {
      if (newToken) {
        localStorage.setItem('access_token', newToken);
      } else {
        localStorage.removeItem('access_token');
      }
    }
  };

  const value: AuthContextType = {
    token,
    isAuthenticated: !!token,
    setToken,
  };

  // Handle hydration by not rendering until mounted
  if (!isMounted) {
    return null;
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
