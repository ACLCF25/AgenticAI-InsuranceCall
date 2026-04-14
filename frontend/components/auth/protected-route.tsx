'use client'

import { useEffect } from 'react'
import { useRouter, usePathname } from 'next/navigation'
import { useAuth } from '@/lib/auth-context'
import { getStoredSession } from '@/lib/supabase-auth'

interface ProtectedRouteProps {
  children: React.ReactNode
}

export function ProtectedRoute({ children }: ProtectedRouteProps) {
  const { user, isLoading } = useAuth()
  const router = useRouter()
  const pathname = usePathname()
  const hasStoredSession = typeof window !== 'undefined' && !!getStoredSession()

  useEffect(() => {
    if (isLoading) return
    if (!user) {
      router.replace('/login')
      return
    }
    if (!user.email_confirmed || user.approval_status !== 'approved') {
      router.replace('/pending-approval')
      return
    }

    if (user.role === 'agent' && !pathname.startsWith('/calls')) {
      router.replace('/calls')
      return
    }

    if (pathname.startsWith('/settings/audit') && user.role !== 'super_admin') {
      router.replace('/')
    }
  }, [user, isLoading, router, pathname])

  const spinner = (
    <div
      className="flex h-screen items-center justify-center bg-background"
      suppressHydrationWarning
    >
      <div className="flex flex-col items-center gap-3" suppressHydrationWarning>
        <div
          className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent"
          suppressHydrationWarning
        />
        <p className="text-sm text-muted-foreground" suppressHydrationWarning>Loading...</p>
      </div>
    </div>
  )

  if (isLoading) return spinner

  if (!user) {
    if (hasStoredSession) {
      return spinner
    }
    return null
  }

  if (!user.email_confirmed || user.approval_status !== 'approved') return null
  if (user.role === 'agent' && !pathname.startsWith('/calls')) return null
  if (pathname.startsWith('/settings/audit') && user.role !== 'super_admin') return null

  return <>{children}</>
}
