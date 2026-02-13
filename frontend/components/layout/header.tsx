'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { Plus, Menu, LogOut } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useState } from 'react'
import { MobileSidebar } from './mobile-sidebar'
import { useAuth } from '@/lib/auth-context'

const pageTitles: Record<string, string> = {
  '/': 'Dashboard',
  '/calls': 'Calls',
  '/calls/new': 'Start New Call',
  '/insurance': 'Insurance Providers',
  '/analytics': 'Analytics',
  '/settings': 'Settings',
}

export function Header() {
  const pathname = usePathname()
  const [mobileOpen, setMobileOpen] = useState(false)
  const { user, logout } = useAuth()

  const getTitle = () => {
    if (pathname.startsWith('/calls/') && pathname !== '/calls/new' && pathname !== '/calls') {
      return 'Call Detail'
    }
    if (pathname.startsWith('/insurance/') && pathname !== '/insurance') {
      return 'Provider Detail'
    }
    return pageTitles[pathname] || 'Dashboard'
  }

  return (
    <>
      <header className="sticky top-0 z-40 flex h-16 items-center gap-3 border-b border-border/70 bg-background/85 px-4 backdrop-blur md:px-8">
        <Button
          variant="ghost"
          size="icon"
          className="h-9 w-9 md:hidden"
          onClick={() => setMobileOpen(true)}
          aria-label="Open menu"
        >
          <Menu className="h-5 w-5" />
        </Button>

        <div>
          <p className="text-[11px] uppercase tracking-[0.08em] text-muted-foreground">Credentialing Agent</p>
          <h1 className="text-base font-semibold tracking-tight md:text-lg">{getTitle()}</h1>
        </div>

        <div className="flex-1" />

        {user?.role === 'admin' && (
          <Button asChild size="sm" className="gap-1.5 rounded-full px-4">
            <Link href="/calls/new">
              <Plus className="h-4 w-4" />
              <span className="hidden sm:inline">New Call</span>
            </Link>
          </Button>
        )}

        <Button
          variant="ghost"
          size="icon"
          className="h-9 w-9"
          onClick={logout}
          aria-label={`Sign out (${user?.username})`}
          title={`Signed in as ${user?.username}`}
        >
          <LogOut className="h-4 w-4" />
        </Button>
      </header>

      <MobileSidebar open={mobileOpen} onClose={() => setMobileOpen(false)} />
    </>
  )
}
