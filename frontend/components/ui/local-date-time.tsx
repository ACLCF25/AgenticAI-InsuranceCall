'use client'

import { useEffect, useState } from 'react'
import { formatDate } from '@/lib/utils'

interface LocalDateTimeProps {
  value?: string | Date | null
  fallback?: string
  className?: string
}

export function LocalDateTime({
  value,
  fallback = '-',
  className,
}: LocalDateTimeProps) {
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  if (!value) {
    return <span className={className}>{fallback}</span>
  }

  return (
    <span className={className} suppressHydrationWarning>
      {mounted ? formatDate(value) : fallback}
    </span>
  )
}
