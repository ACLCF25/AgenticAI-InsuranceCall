// app/page.tsx
'use client'

import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api'
import { DashboardHeader } from '@/components/dashboard/header'
import { StatsCards } from '@/components/dashboard/stats-cards'
import { ActiveCallsTable } from '@/components/dashboard/active-calls-table'
import { RecentCallsTable } from '@/components/dashboard/recent-calls-table'
import { MetricsChart } from '@/components/dashboard/metrics-chart'
import { FollowupsPanel } from '@/components/dashboard/followups-panel'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Skeleton } from '@/components/ui/skeleton'

export default function DashboardPage() {
  const { data: metrics, isLoading: metricsLoading } = useQuery({
    queryKey: ['metrics'],
    queryFn: () => api.getMetrics(),
    refetchInterval: 30000, // Refresh every 30 seconds
  })

  const { data: followups, isLoading: followupsLoading } = useQuery({
    queryKey: ['followups'],
    queryFn: () => api.getScheduledFollowups(),
    refetchInterval: 60000, // Refresh every minute
  })

  return (
    <div className="min-h-screen bg-background">
      <DashboardHeader />
      
      <main className="container mx-auto py-6 px-4">
        {/* Stats Overview */}
        <div className="mb-8">
          {metricsLoading ? (
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
              {[...Array(4)].map((_, i) => (
                <Skeleton key={i} className="h-32" />
              ))}
            </div>
          ) : (
            <StatsCards metrics={metrics} />
          )}
        </div>

        {/* Main Content Tabs */}
        <Tabs defaultValue="overview" className="space-y-6">
          <TabsList className="grid w-full grid-cols-4 lg:w-auto">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="calls">Calls</TabsTrigger>
            <TabsTrigger value="followups">Follow-ups</TabsTrigger>
            <TabsTrigger value="analytics">Analytics</TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-6">
            <div className="grid gap-6 lg:grid-cols-2">
              <ActiveCallsTable />
              <FollowupsPanel followups={followups?.data || []} />
            </div>
            <RecentCallsTable />
          </TabsContent>

          {/* Calls Tab */}
          <TabsContent value="calls" className="space-y-6">
            <RecentCallsTable showAll />
          </TabsContent>

          {/* Follow-ups Tab */}
          <TabsContent value="followups" className="space-y-6">
            <FollowupsPanel 
              followups={followups?.data || []} 
              showAll 
            />
          </TabsContent>

          {/* Analytics Tab */}
          <TabsContent value="analytics" className="space-y-6">
            <MetricsChart />
          </TabsContent>
        </Tabs>
      </main>
    </div>
  )
}
