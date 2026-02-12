'use client'

import { CheckCircle, Clock, Phone, TrendingUp, type LucideIcon } from 'lucide-react'
import { motion } from 'framer-motion'
import { Card, CardContent } from '@/components/ui/card'
import type { SystemMetrics } from '@/types'

interface StatsCardsProps {
  metrics?: SystemMetrics
}

interface StatItem {
  title: string
  value: string | number
  icon: LucideIcon
  description: string
  iconClass: string
}

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.08 },
  },
}

const item = {
  hidden: { opacity: 0, y: 10 },
  show: { opacity: 1, y: 0 },
}

export function StatsCards({ metrics }: StatsCardsProps) {
  const stats: StatItem[] = [
    {
      title: 'Total Calls',
      value: metrics?.total_calls || 0,
      icon: Phone,
      description: `Last ${metrics?.period_days || 7} days`,
      iconClass: 'text-primary',
    },
    {
      title: 'Approved',
      value: metrics?.approved || 0,
      icon: CheckCircle,
      description: 'Successfully completed',
      iconClass: 'text-emerald-600 dark:text-emerald-400',
    },
    {
      title: 'In Progress',
      value: metrics?.in_progress || 0,
      icon: Clock,
      description: 'Pending and under review',
      iconClass: 'text-amber-600 dark:text-amber-400',
    },
    {
      title: 'Success Rate',
      value: `${metrics?.success_rate || 0}%`,
      icon: TrendingUp,
      description: 'Approval rate',
      iconClass: 'text-sky-600 dark:text-sky-400',
    },
  ]

  return (
    <motion.div
      variants={container}
      initial="hidden"
      animate="show"
      className="grid gap-4 md:grid-cols-2 lg:grid-cols-4"
    >
      {stats.map((stat) => {
        const Icon = stat.icon
        return (
          <motion.div key={stat.title} variants={item}>
            <Card>
              <CardContent className="p-5">
                <div className="flex items-start justify-between">
                  <div className="space-y-1">
                    <p className="text-[11px] uppercase tracking-[0.08em] text-muted-foreground">{stat.title}</p>
                    <p className="text-2xl font-semibold tracking-tight">{stat.value}</p>
                    <p className="text-xs text-muted-foreground">{stat.description}</p>
                  </div>
                  <div className="rounded-xl bg-muted p-2.5">
                    <Icon className={`h-4 w-4 ${stat.iconClass}`} />
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )
      })}
    </motion.div>
  )
}
