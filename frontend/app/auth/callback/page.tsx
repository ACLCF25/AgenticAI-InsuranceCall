'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { Loader2 } from 'lucide-react'
import { useAuth, getDefaultPathForUser } from '@/lib/auth-context'

export default function AuthCallbackPage() {
  const router = useRouter()
  const { user, isLoading } = useAuth()

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
    router.replace(getDefaultPathForUser(user))
  }, [isLoading, router, user])

  return (
    <div className="flex min-h-screen items-center justify-center bg-background">
      <div className="flex flex-col items-center gap-3">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
        <p className="text-sm text-muted-foreground">Completing sign-in...</p>
      </div>
    </div>
  )
}
