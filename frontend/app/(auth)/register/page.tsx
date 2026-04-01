'use client'

import Link from 'next/link'
import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { UserPlus } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { useAuth } from '@/lib/auth-context'

export default function RegisterPage() {
  const router = useRouter()
  const { register } = useAuth()
  const [username, setUsername] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [serverError, setServerError] = useState<string | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)

  const onSubmit = async (event: React.FormEvent) => {
    event.preventDefault()
    setServerError(null)
    setIsSubmitting(true)
    try {
      await register({ username, email, password })
      router.push(`/check-email?email=${encodeURIComponent(email)}`)
    } catch (err: unknown) {
      setServerError(err instanceof Error ? err.message : 'Registration failed. Please try again.')
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-background p-4">
      <Card className="w-full max-w-sm shadow-lg">
        <CardHeader className="space-y-1 text-center">
          <div className="mb-2 flex justify-center">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10 text-primary">
              <UserPlus className="h-5 w-5" />
            </div>
          </div>
          <CardTitle className="text-xl">Register</CardTitle>
          <CardDescription>Create an agent account for approval</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={onSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="username">Username</Label>
              <Input id="username" value={username} onChange={(e) => setUsername(e.target.value)} placeholder="agent.jane" />
            </div>
            <div className="space-y-2">
              <Label htmlFor="email">Email</Label>
              <Input id="email" type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="jane@example.com" />
            </div>
            <div className="space-y-2">
              <Label htmlFor="password">Password</Label>
              <Input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Minimum 8 characters"
              />
            </div>

            {serverError && (
              <p className="rounded-md bg-destructive/10 px-3 py-2 text-sm font-medium text-destructive">
                {serverError}
              </p>
            )}

            <Button
              type="submit"
              className="w-full"
              disabled={isSubmitting || !username || !email || password.length < 8}
            >
              {isSubmitting ? 'Creating account...' : 'Create Account'}
            </Button>
          </form>

          <div className="mt-4 text-center text-sm text-muted-foreground">
            Already registered?{' '}
            <Link href="/login" className="font-medium text-primary hover:underline">
              Sign in
            </Link>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
