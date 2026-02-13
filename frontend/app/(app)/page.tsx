'use client'

import { useQuery } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import { ArrowRight, Plus } from 'lucide-react'
import { api } from '@/lib/api'
import { StatsCards } from '@/components/dashboard/stats-cards'
import { ActiveCallsTable } from '@/components/dashboard/active-calls-table'
import { FollowupsPanel } from '@/components/dashboard/followups-panel'
import { Skeleton } from '@/components/ui/skeleton'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { formatRelativeTime, formatStatus, getStatusColor } from '@/lib/utils'

export default function DashboardPage() {
  const router = useRouter()

  const { data: metrics, isLoading: metricsLoading } = useQuery({
    queryKey: ['metrics'],
    queryFn: () => api.getMetrics(),
    refetchInterval: 30000,
  })

  const { data: followups } = useQuery({
    queryKey: ['followups'],
    queryFn: () => api.getScheduledFollowups(),
    refetchInterval: 60000,
  })

  const { data: recentCalls } = useQuery({
    queryKey: ['recent-calls', 5],
    queryFn: () => api.getRecentCalls(5),
    refetchInterval: 30000,
  })

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="space-y-6"
    >
      <Card className="border-dashed border-border/80">
        <CardContent className="flex flex-col gap-4 pt-5 md:flex-row md:items-center md:justify-between">
          <div>
            <p className="text-[11px] uppercase tracking-[0.08em] text-muted-foreground">Overview</p>
            <h2 className="mt-1 text-2xl font-semibold tracking-tight">Operations Dashboard</h2>
            <p className="mt-1 text-sm text-muted-foreground">
              Monitor call progress, pending actions, and recent activity.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button asChild size="sm" className="rounded-full px-4">
              <Link href="/calls/new">
                <Plus className="h-4 w-4" />
                Start Call
              </Link>
            </Button>
            <Button asChild variant="outline" size="sm" className="rounded-full px-4">
              <Link href="/calls">
                Call History
              </Link>
            </Button>
          </div>
        </CardContent>
      </Card>

      {metricsLoading ? (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          {[...Array(4)].map((_, i) => (
            <Skeleton key={i} className="h-28 rounded-2xl" />
          ))}
        </div>
      ) : (
        <StatsCards metrics={metrics} />
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        <ActiveCallsTable />
        <FollowupsPanel followups={followups?.data || []} />
      </div>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-base font-semibold">Recent Calls</CardTitle>
          <Button asChild variant="ghost" size="sm" className="gap-1 text-xs">
            <Link href="/calls">
              View all
              <ArrowRight className="h-3 w-3" />
            </Link>
          </Button>
        </CardHeader>
        <CardContent>
          {recentCalls?.data && recentCalls.data.length > 0 ? (
            <div className="space-y-2">
              {recentCalls.data.map((call) => (
                <div
                  key={call.id}
                  className="flex cursor-pointer items-center justify-between rounded-xl border border-border/60 p-3 transition-colors hover:bg-muted/40"
                  onClick={() => call.id && router.push(`/calls/${call.id}`)}
                >
                  <div className="min-w-0">
                    <p className="truncate text-sm font-medium">{call.provider_name}</p>
                    <p className="truncate text-xs text-muted-foreground">{call.insurance_name}</p>
                  </div>
                  <div className="flex flex-shrink-0 items-center gap-3">
                    <Badge className={getStatusColor(call.status || 'initiated')}>
                      {formatStatus(call.status || 'initiated')}
                    </Badge>
                    <span className="hidden text-xs text-muted-foreground sm:inline">
                      {call.created_at ? formatRelativeTime(call.created_at) : '-'}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="py-6 text-center text-sm text-muted-foreground">
              No recent calls
            </p>
          )}
        </CardContent>
      </Card>
    </motion.div>
  )
}
