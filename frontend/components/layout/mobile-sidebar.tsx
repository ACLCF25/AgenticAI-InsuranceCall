'use client'

import { useEffect } from 'react'
import { X, Bot } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { SidebarNav } from './sidebar-nav'
import { ThemeSwitcher } from './theme-switcher'
import { motion, AnimatePresence } from 'framer-motion'

interface MobileSidebarProps {
  open: boolean
  onClose: () => void
}

export function MobileSidebar({ open, onClose }: MobileSidebarProps) {
  useEffect(() => {
    if (open) {
      document.body.style.overflow = 'hidden'
    } else {
      document.body.style.overflow = ''
    }
    return () => {
      document.body.style.overflow = ''
    }
  }, [open])

  return (
    <AnimatePresence>
      {open && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 bg-black/45 md:hidden"
            onClick={onClose}
          />

          <motion.aside
            initial={{ x: '-100%' }}
            animate={{ x: 0 }}
            exit={{ x: '-100%' }}
            transition={{ type: 'spring', bounce: 0, duration: 0.3 }}
            className="fixed inset-y-0 left-0 z-50 w-[260px] border-r border-sidebar-border bg-sidebar/95 backdrop-blur md:hidden"
          >
            <div className="flex h-16 items-center justify-between border-b border-sidebar-border px-5">
              <div className="flex items-center gap-3">
                <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-primary/10 text-primary">
                  <Bot className="h-4 w-4" />
                </div>
                <span className="text-sm font-semibold">Credentialing</span>
              </div>
              <Button variant="ghost" size="icon" className="h-9 w-9" onClick={onClose} aria-label="Close menu">
                <X className="h-4 w-4" />
              </Button>
            </div>

            <div className="flex-1 overflow-y-auto py-4" onClick={onClose}>
              <SidebarNav />
            </div>

            <div className="border-t border-sidebar-border px-4 py-3">
              <div className="flex items-center justify-between">
                <span className="text-[11px] uppercase tracking-[0.08em] text-muted-foreground">v1.0.0</span>
                <ThemeSwitcher />
              </div>
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  )
}
