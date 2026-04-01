'use client'

import { createContext, useContext, useEffect, useState, ReactNode } from 'react'
import { useRouter } from 'next/navigation'
import { api } from './api'
import type { AuthUser } from '@/types'
import {
  clearStoredSession,
  getStoredSession,
  parseSessionFromHash,
  refreshSession,
  sendPasswordReset,
  signInWithPassword,
  signOut,
  signUpWithPassword,
  storeSession,
  type StoredSession,
  updatePassword,
} from './supabase-auth'

interface AuthContextValue {
  user: AuthUser | null
  token: string | null
  isLoading: boolean
  login: (email: string, password: string) => Promise<AuthUser>
  register: (input: { username: string; email: string; password: string }) => Promise<void>
  logout: () => Promise<void>
  requestPasswordReset: (email: string) => Promise<void>
  resetPassword: (password: string) => Promise<void>
  refreshProfile: () => Promise<AuthUser | null>
}

const AuthContext = createContext<AuthContextValue | null>(null)

function getApiUrl() {
  return process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000/api'
}

async function fetchProfile(accessToken: string): Promise<AuthUser> {
  const response = await fetch(`${getApiUrl()}/auth/me`, {
    headers: {
      Authorization: `Bearer ${accessToken}`,
    },
  })

  const data = await response.json().catch(() => ({}))
  if (!response.ok) {
    throw new Error(data.error || 'Failed to load user profile')
  }

  return data.user
}

export function getDefaultPathForUser(user: Pick<AuthUser, 'role'>) {
  return user.role === 'agent' ? '/calls' : '/'
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null)
  const [session, setSession] = useState<StoredSession | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const router = useRouter()

  const clearAuthState = () => {
    clearStoredSession()
    api.setAuthToken(null)
    setSession(null)
    setUser(null)
  }

  const syncProfile = async (nextSession: StoredSession | null): Promise<AuthUser | null> => {
    if (!nextSession) {
      clearAuthState()
      return null
    }

    storeSession(nextSession)
    setSession(nextSession)
    api.setAuthToken(nextSession.access_token)

    try {
      const profile = await fetchProfile(nextSession.access_token)
      setUser(profile)
      return profile
    } catch (error) {
      const refreshToken = nextSession.refresh_token
      if (refreshToken) {
        try {
          const refreshed = await refreshSession(refreshToken)
          setSession(refreshed)
          api.setAuthToken(refreshed.access_token)
          const profile = await fetchProfile(refreshed.access_token)
          setUser(profile)
          return profile
        } catch {
          clearAuthState()
          throw error
        }
      }

      clearAuthState()
      throw error
    }
  }

  useEffect(() => {
    let cancelled = false

    const initialise = async () => {
      try {
        let nextSession = getStoredSession()

        if (typeof window !== 'undefined') {
          const fromHash = parseSessionFromHash(window.location.hash)
          if (fromHash) {
            nextSession = fromHash
            storeSession(fromHash)
            window.history.replaceState(null, '', `${window.location.pathname}${window.location.search}`)
          }
        }

        if (!cancelled) {
          if (nextSession) {
            await syncProfile(nextSession)
          } else {
            clearAuthState()
          }
        }
      } catch {
        if (!cancelled) {
          clearAuthState()
        }
      } finally {
        if (!cancelled) {
          setIsLoading(false)
        }
      }
    }

    void initialise()

    return () => {
      cancelled = true
    }
  }, [])

  const login = async (email: string, password: string) => {
    const nextSession = await signInWithPassword(email, password)
    const profile = await syncProfile(nextSession)
    if (!profile) {
      throw new Error('Failed to load account profile')
    }
    return profile
  }

  const register = async (input: { username: string; email: string; password: string }) => {
    await signUpWithPassword(input)
  }

  const logout = async () => {
    try {
      await signOut(session?.access_token)
    } finally {
      clearAuthState()
      router.push('/login')
    }
  }

  const requestPasswordResetHandler = async (email: string) => {
    await sendPasswordReset(email)
  }

  const resetPasswordHandler = async (password: string) => {
    const currentSession = session || getStoredSession()
    if (!currentSession?.access_token) {
      throw new Error('Reset session is missing or expired')
    }

    await updatePassword(currentSession.access_token, password)
  }

  const refreshProfile = async () => {
    const currentSession = session || getStoredSession()
    if (!currentSession) {
      clearAuthState()
      return null
    }

    return syncProfile(currentSession)
  }

  return (
    <AuthContext.Provider
      value={{
        user,
        token: session?.access_token || null,
        isLoading,
        login,
        register,
        logout,
        requestPasswordReset: requestPasswordResetHandler,
        resetPassword: resetPasswordHandler,
        refreshProfile,
      }}
    >
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used within AuthProvider')
  return ctx
}
