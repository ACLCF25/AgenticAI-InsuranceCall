'use client'

import { useQuery } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import { BarChart3, Loader2, TrendingUp, Phone, CheckCircle, Clock } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { api } from '@/lib/api'

export default function AnalyticsPage() {
  const { data: metrics, isLoading } = useQuery({
    queryKey: ['metrics'],
    queryFn: () => api.getMetrics(),
    refetchInterval: 60000,
  })

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    )
  }

  const stats = [
    {
      title: 'Total Calls',
      value: metrics?.total_calls || 0,
      icon: Phone,
      color: 'text-blue-400',
      bg: 'bg-blue-500/15',
    },
    {
      title: 'Approved',
      value: metrics?.approved || 0,
      icon: CheckCircle,
      color: 'text-green-400',
      bg: 'bg-green-500/15',
    },
    {
      title: 'In Progress',
      value: metrics?.in_progress || 0,
      icon: Clock,
      color: 'text-yellow-400',
      bg: 'bg-yellow-500/15',
    },
    {
      title: 'Success Rate',
      value: `${metrics?.success_rate || 0}%`,
      icon: TrendingUp,
      color: 'text-purple-400',
      bg: 'bg-purple-500/15',
    },
  ]

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      {/* Metrics overview */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {stats.map((stat) => {
          const Icon = stat.icon
          return (
            <Card key={stat.title}>
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-xs text-muted-foreground font-medium">{stat.title}</p>
                    <p className="text-2xl font-bold mt-1">{stat.value}</p>
                  </div>
                  <div className={`rounded-lg p-2.5 ${stat.bg}`}>
                    <Icon className={`h-4 w-4 ${stat.color}`} />
                  </div>
                </div>
              </CardContent>
            </Card>
          )
        })}
      </div>

      {/* Charts placeholder */}
      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4 text-muted-foreground" />
              <CardTitle className="text-base font-semibold">Weekly Overview</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <div className="flex flex-col items-center justify-center py-16 text-center">
              <BarChart3 className="h-10 w-10 text-muted-foreground mb-3" />
              <p className="text-sm text-muted-foreground">
                Charts will be wired to real API data
              </p>
              <p className="text-xs text-muted-foreground mt-1">
                Period: Last {metrics?.period_days || 7} days
              </p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base font-semibold">Status Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm">Approved</span>
                <span className="text-sm font-medium text-green-400">{metrics?.approved || 0}</span>
              </div>
              <div className="w-full bg-muted rounded-full h-2">
                <div
                  className="bg-green-500 h-2 rounded-full transition-all"
                  style={{ width: `${metrics?.total_calls ? (metrics.approved / metrics.total_calls) * 100 : 0}%` }}
                />
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">In Progress</span>
                <span className="text-sm font-medium text-yellow-400">{metrics?.in_progress || 0}</span>
              </div>
              <div className="w-full bg-muted rounded-full h-2">
                <div
                  className="bg-yellow-500 h-2 rounded-full transition-all"
                  style={{ width: `${metrics?.total_calls ? (metrics.in_progress / metrics.total_calls) * 100 : 0}%` }}
                />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </motion.div>
  )
}
