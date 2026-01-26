'use client';

import React, { createContext, useContext, useState, useEffect } from 'react';

interface AuthContextType {
  token: string | null;
  isAuthenticated: boolean;
  setToken: (token: string | null) => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [token, setTokenState] = useState<string | null>(null);
  const [isMounted, setIsMounted] = useState(false);

  // Load token from localStorage on mount
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const storedToken = localStorage.getItem('access_token');
      setTokenState(storedToken);
    }
    setIsMounted(true);
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
