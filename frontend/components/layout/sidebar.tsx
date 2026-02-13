'use client'

import { Bot } from 'lucide-react'
import { SidebarNav } from './sidebar-nav'
import { ThemeSwitcher } from './theme-switcher'
import { useAuth } from '@/lib/auth-context'

export function Sidebar() {
  const { user } = useAuth()

  return (
    <aside className="hidden md:fixed md:inset-y-4 md:left-4 md:z-40 md:flex md:w-[220px] md:flex-col rounded-2xl border border-sidebar-border bg-sidebar/95 backdrop-blur">
      <div className="flex items-center gap-3 border-b border-sidebar-border px-5 py-4">
        <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-primary/10 text-primary">
          <Bot className="h-4 w-4" />
        </div>
        <div className="flex flex-col">
          <span className="text-sm font-semibold tracking-tight">Credentialing</span>
          <span className="text-[11px] text-muted-foreground leading-none">Control Panel</span>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto py-4">
        <SidebarNav />
      </div>

      <div className="border-t border-sidebar-border px-4 py-3">
        <div className="flex items-center justify-between">
          <div className="flex flex-col">
            <span className="text-xs font-medium">{user?.username}</span>
            <span className="text-[11px] capitalize text-muted-foreground">{user?.role}</span>
          </div>
          <ThemeSwitcher />
        </div>
      </div>
    </aside>
  )
}
