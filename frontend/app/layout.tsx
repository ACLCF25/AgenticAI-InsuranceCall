import type { Metadata } from 'next'
import { Manrope } from 'next/font/google'
import './globals.css'
import { Providers } from './providers'
import { Toaster } from '@/components/ui/toaster'

const manrope = Manrope({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Autonomous Credentialing Agent',
  description: 'AI-powered insurance credentialing automation system',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${manrope.className} antialiased`}>
        <Providers>
          {children}
          <Toaster />
        </Providers>
      </body>
    </html>
  )
}
