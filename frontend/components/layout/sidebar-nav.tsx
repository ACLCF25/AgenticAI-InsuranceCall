'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { motion } from 'framer-motion'
import {
  LayoutDashboard,
  Phone,
  Building2,
  BarChart3,
  Settings,
  Users,
  PhoneCall,
  type LucideIcon,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { useAuth } from '@/lib/auth-context'

interface NavItem {
  href: string
  label: string
  icon: LucideIcon
}

const adminNavItems: NavItem[] = [
  { href: '/', label: 'Dashboard', icon: LayoutDashboard },
  { href: '/calls', label: 'Calls', icon: Phone },
  { href: '/insurance', label: 'Insurance', icon: Building2 },
  { href: '/analytics', label: 'Analytics', icon: BarChart3 },
  { href: '/settings', label: 'Settings', icon: Settings },
  { href: '/settings/users', label: 'Users', icon: Users },
  { href: '/settings/phone-numbers', label: 'Phone Lines', icon: PhoneCall },
]

const superAdminNavItems: NavItem[] = [
  ...adminNavItems,
  { href: '/settings/audit', label: 'Audit Trail', icon: BarChart3 },
]

const agentNavItems: NavItem[] = [
  { href: '/calls', label: 'My Calls', icon: Phone },
  { href: '/calls/new', label: 'New Call', icon: PhoneCall },
]

export function SidebarNav() {
  const pathname = usePathname()
  const { user } = useAuth()
  const navItems =
    user?.role === 'super_admin'
      ? superAdminNavItems
      : user?.role === 'admin'
        ? adminNavItems
        : agentNavItems

  const isActive = (href: string) => {
    if (href === '/') return pathname === '/'
    if (href === '/settings') return pathname === '/settings'
    if (href === '/calls') return pathname === '/calls' || (pathname.startsWith('/calls/') && pathname !== '/calls/new')
    return pathname.startsWith(href)
  }

  return (
    <nav className="flex flex-col gap-1 px-3">
      {navItems.map((item) => {
        const Icon = item.icon
        const active = isActive(item.href)

        return (
          <Link
            key={item.href}
            href={item.href}
            className={cn(
              'relative flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition-colors',
              active
                ? 'text-foreground'
                : 'text-muted-foreground hover:bg-sidebar-accent hover:text-foreground'
            )}
          >
            {active && (
              <motion.div
                layoutId="sidebar-active"
                className="absolute inset-0 rounded-xl bg-sidebar-accent"
                transition={{ type: 'spring', bounce: 0.2, duration: 0.4 }}
              />
            )}
            <Icon className="relative z-10 h-4 w-4" />
            <span className="relative z-10">{item.label}</span>
            {active && (
              <motion.div
                layoutId="sidebar-indicator"
                className="absolute right-2.5 top-1/2 h-1.5 w-1.5 -translate-y-1/2 rounded-full bg-primary"
                transition={{ type: 'spring', bounce: 0.2, duration: 0.4 }}
              />
            )}
          </Link>
        )
      })}
    </nav>
  )
}
