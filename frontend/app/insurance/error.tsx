'use client'

import { AlertTriangle } from 'lucide-react'
import { Button } from '@/components/ui/button'

export default function InsuranceError({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  return (
    <div className="flex flex-col items-center justify-center py-20 text-center">
      <AlertTriangle className="h-10 w-10 text-muted-foreground mb-4" />
      <p className="text-lg font-medium">Failed to load insurance providers</p>
      <p className="text-sm text-muted-foreground mt-1">{error.message}</p>
      <Button variant="outline" onClick={reset} className="mt-6">
        Try again
      </Button>
    </div>
  )
}
