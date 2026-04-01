'use client'

import { Suspense } from 'react'
import Link from 'next/link'
import { useSearchParams } from 'next/navigation'
import { MailCheck } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'

function CheckEmailContent() {
  const searchParams = useSearchParams()
  const email = searchParams.get('email')

  return (
    <div className="flex min-h-screen items-center justify-center bg-background p-4">
      <Card className="w-full max-w-md shadow-lg">
        <CardHeader className="space-y-2 text-center">
          <div className="mx-auto flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10 text-primary">
            <MailCheck className="h-5 w-5" />
          </div>
          <CardTitle>Check your email</CardTitle>
          <CardDescription>
            We sent a confirmation link{email ? ` to ${email}` : ''}. Confirm your email first, then an admin can approve your account.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4 text-center">
          <p className="text-sm text-muted-foreground">
            After verification, sign in and the app will tell you whether your account is still waiting for admin approval.
          </p>
          <Button asChild className="w-full">
            <Link href="/login">Back to login</Link>
          </Button>
        </CardContent>
      </Card>
    </div>
  )
}

export default function CheckEmailPage() {
  return (
    <Suspense
      fallback={
        <div className="flex min-h-screen items-center justify-center bg-background p-4">
          <Card className="w-full max-w-md shadow-lg">
            <CardHeader className="space-y-2 text-center">
              <div className="mx-auto flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10 text-primary">
                <MailCheck className="h-5 w-5" />
              </div>
              <CardTitle>Check your email</CardTitle>
              <CardDescription>Loading confirmation details...</CardDescription>
            </CardHeader>
          </Card>
        </div>
      }
    >
      <CheckEmailContent />
    </Suspense>
  )
}
