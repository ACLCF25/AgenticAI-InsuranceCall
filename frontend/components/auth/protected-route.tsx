'use client'

import { useEffect } from 'react'
import { useRouter, usePathname } from 'next/navigation'
import { useAuth } from '@/lib/auth-context'

interface ProtectedRouteProps {
  children: React.ReactNode
}

export function ProtectedRoute({ children }: ProtectedRouteProps) {
  const { user, isLoading } = useAuth()
  const router = useRouter()
  const pathname = usePathname()

  useEffect(() => {
    if (isLoading) return
    if (!user) {
      router.replace('/login')
      return
    }
    // role=user can only access /calls/new
    if (user.role === 'user' && pathname !== '/calls/new') {
      router.replace('/calls/new')
    }
  }, [user, isLoading, router, pathname])

  const spinner = (
    <div className="flex h-screen items-center justify-center bg-background">
      <div className="flex flex-col items-center gap-3">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
        <p className="text-sm text-muted-foreground">Loading...</p>
      </div>
    </div>
  )

  if (isLoading) return spinner

  // Token exists in localStorage but AuthProvider hasn't propagated user state yet
  // (race condition right after login). Hold on loading instead of redirecting.
  if (!user) {
    if (typeof window !== 'undefined' && localStorage.getItem('auth_token')) {
      return spinner
    }
    return null
  }

  if (user.role === 'user' && pathname !== '/calls/new') return null

  return <>{children}</>
}
