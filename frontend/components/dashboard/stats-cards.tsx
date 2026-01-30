// components/dashboard/stats-cards.tsx
'use client'

import { Phone, CheckCircle, Clock, TrendingUp } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import type { SystemMetrics } from '@/types'

interface StatsCardsProps {
  metrics?: SystemMetrics
}

export function StatsCards({ metrics }: StatsCardsProps) {
  const stats = [
    {
      title: 'Total Calls',
      value: metrics?.total_calls || 0,
      icon: Phone,
      description: `Last ${metrics?.period_days || 7} days`,
      color: 'text-blue-600',
      bgColor: 'bg-blue-100',
    },
    {
      title: 'Approved',
      value: metrics?.approved || 0,
      icon: CheckCircle,
      description: 'Successfully completed',
      color: 'text-green-600',
      bgColor: 'bg-green-100',
    },
    {
      title: 'In Progress',
      value: metrics?.in_progress || 0,
      icon: Clock,
      description: 'Pending & under review',
      color: 'text-yellow-600',
      bgColor: 'bg-yellow-100',
    },
    {
      title: 'Success Rate',
      value: `${metrics?.success_rate || 0}%`,
      icon: TrendingUp,
      description: 'Approval rate',
      color: 'text-purple-600',
      bgColor: 'bg-purple-100',
    },
  ]

  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
      {stats.map((stat) => {
        const Icon = stat.icon
        return (
          <Card key={stat.title}>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">
                {stat.title}
              </CardTitle>
              <div className={`rounded-full p-2 ${stat.bgColor}`}>
                <Icon className={`h-4 w-4 ${stat.color}`} />
              </div>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stat.value}</div>
              <p className="text-xs text-muted-foreground">
                {stat.description}
              </p>
            </CardContent>
          </Card>
        )
      })}
    </div>
  )
}
