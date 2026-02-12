'use client'

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ThemeProvider } from 'next-themes'
import { useState } from 'react'

// Suppress React 19 ref warnings from Radix UI components
if (typeof window !== 'undefined') {
  const originalError = console.error
  console.error = (...args: unknown[]) => {
    const message = args[0]
    if (typeof message === 'string' && message.includes('Accessing element.ref')) {
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
        {children}
      </QueryClientProvider>
    </ThemeProvider>
  )
}
