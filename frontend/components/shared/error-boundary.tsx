'use client'

import { Component, type ReactNode } from 'react'
import { AlertTriangle } from 'lucide-react'
import { Button } from '@/components/ui/button'

interface Props {
  children: ReactNode
  fallback?: ReactNode
}

interface State {
  hasError: boolean
  error?: Error
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback || (
        <div className="flex flex-col items-center justify-center py-20 text-center">
          <AlertTriangle className="h-10 w-10 text-muted-foreground mb-4" />
          <p className="text-lg font-medium">Something went wrong</p>
          <p className="text-sm text-muted-foreground mt-1">{this.state.error?.message}</p>
          <Button
            variant="outline"
            onClick={() => this.setState({ hasError: false })}
            className="mt-4"
          >
            Try again
          </Button>
        </div>
      )
    }
    return this.props.children
  }
}
