'use client'

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ThemeProvider } from 'next-themes'
import { useState } from 'react'
import { AuthProvider } from '@/lib/auth-context'

// Suppress React 19 ref warnings from Radix UI components
if (typeof window !== 'undefined') {
  const originalError = console.error
  console.error = (...args: unknown[]) => {
    const message = args.find((arg) => typeof arg === 'string') as string | undefined
    const joined = args
      .filter((arg) => typeof arg === 'string')
      .join(' ')

    if (message?.includes('Accessing element.ref')) {
      return
    }

    // Ignore extension-injected hydration noise (for example, bis_skin_checked).
    if (joined.includes('bis_skin_checked')) {
      return
    }
    originalError.apply(console, args)
  }
}

export function Providers({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 60 * 1000,
            refetchOnWindowFocus: false,
          },
        },
      })
  )

  return (
    <ThemeProvider
      attribute="class"
      defaultTheme="light"
      enableSystem={false}
      storageKey="credentialing-theme"
      themes={['dark', 'light']}
    >
      <QueryClientProvider client={queryClient}>
        <AuthProvider>
          {children}
        </AuthProvider>
      </QueryClientProvider>
    </ThemeProvider>
  )
}
