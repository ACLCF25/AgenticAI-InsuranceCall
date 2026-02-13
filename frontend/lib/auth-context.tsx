'use client'

import { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import { useRouter } from 'next/navigation'
import { api } from './api'

export interface User {
  id: string
  username: string
  email: string
  role: 'admin' | 'user'
}

interface AuthContextValue {
  user: User | null
  token: string | null
  isLoading: boolean
  login: (username: string, password: string) => Promise<void>
  logout: () => Promise<void>
}

const AuthContext = createContext<AuthContextValue | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [token, setToken] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const router = useRouter()

  // Restore session from localStorage on mount
  useEffect(() => {
    try {
      const storedToken = localStorage.getItem('auth_token')
      const storedUser = localStorage.getItem('auth_user')
      if (storedToken && storedUser) {
        api.setAuthToken(storedToken)
        setToken(storedToken)
        setUser(JSON.parse(storedUser))
      }
    } catch {
      // Ignore parse errors – treat as unauthenticated
    } finally {
      setIsLoading(false)
    }
  }, [])

  const login = async (username: string, password: string) => {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000/api'
    const res = await fetch(`${apiUrl}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    })

    const data = await res.json()

    if (!res.ok) {
      throw new Error(data.error || 'Login failed')
    }

    localStorage.setItem('auth_token', data.access_token)
    localStorage.setItem('auth_refresh_token', data.refresh_token)
    localStorage.setItem('auth_user', JSON.stringify(data.user))

    api.setAuthToken(data.access_token)
    setToken(data.access_token)
    setUser(data.user)

    // Use full navigation so AuthProvider re-reads from localStorage cleanly
    window.location.href = data.user.role === 'admin' ? '/' : '/calls/new'
  }

  const logout = async () => {
    try {
      const currentToken = localStorage.getItem('auth_token')
      if (currentToken) {
        const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000/api'
        await fetch(`${apiUrl}/auth/logout`, {
          method: 'POST',
          headers: { Authorization: `Bearer ${currentToken}` },
        })
      }
    } catch {
      // Fail silently – always clear local state
    } finally {
      localStorage.removeItem('auth_token')
      localStorage.removeItem('auth_refresh_token')
      localStorage.removeItem('auth_user')
      api.setAuthToken(null)
      setToken(null)
      setUser(null)
      router.push('/login')
    }
  }

  return (
    <AuthContext.Provider value={{ user, token, isLoading, login, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used within AuthProvider')
  return ctx
}
