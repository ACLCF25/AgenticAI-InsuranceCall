'use client'

import { useEffect } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { Clock3, Mail, ShieldCheck } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { useAuth, getDefaultPathForUser } from '@/lib/auth-context'

export default function PendingApprovalPage() {
  const router = useRouter()
  const { user, isLoading, logout } = useAuth()

  useEffect(() => {
    if (isLoading) return
    if (!user) {
      router.replace('/login')
      return
    }
    if (user.email_confirmed && user.approval_status === 'approved') {
      router.replace(getDefaultPathForUser(user))
    }
  }, [isLoading, router, user])

  const waitingOnEmail = !user?.email_confirmed

  return (
    <div className="flex min-h-screen items-center justify-center bg-background p-4">
      <Card className="w-full max-w-md shadow-lg">
        <CardHeader className="space-y-2 text-center">
          <div className="mx-auto flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10 text-primary">
            {waitingOnEmail ? <Mail className="h-5 w-5" /> : <Clock3 className="h-5 w-5" />}
          </div>
          <CardTitle>{waitingOnEmail ? 'Confirm your email' : 'Awaiting approval'}</CardTitle>
          <CardDescription>
            {waitingOnEmail
              ? 'Your account exists, but email confirmation must finish before it can be reviewed.'
              : 'Your email is confirmed. An admin still needs to approve your account before you can use the app.'}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="rounded-lg border border-border/60 bg-muted/30 p-4 text-sm text-muted-foreground">
            <p className="font-medium text-foreground">{user?.email || 'Pending account'}</p>
            <p className="mt-1">
              {waitingOnEmail
                ? 'Open the verification email from Supabase Auth, then return here and sign in again.'
                : 'Once approved, you will automatically gain access based on your assigned role.'}
            </p>
          </div>

          <div className="rounded-lg border border-border/60 p-4 text-sm text-muted-foreground">
            <div className="flex items-center gap-2 font-medium text-foreground">
              <ShieldCheck className="h-4 w-4 text-primary" />
              Access model
            </div>
            <p className="mt-2">Agents can start calls and view only their own initiated calls.</p>
            <p>Admins approve agents. Super admins can also view audit history.</p>
          </div>

          <div className="flex gap-3">
            <Button variant="outline" className="flex-1" asChild>
              <Link href="/login">Back to login</Link>
            </Button>
            <Button className="flex-1" onClick={() => void logout()}>
              Sign out
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
