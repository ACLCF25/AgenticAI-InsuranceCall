// components/dashboard/active-calls-table.tsx
'use client'

import { useQuery } from '@tanstack/react-query'
import { Phone, Loader2 } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { api } from '@/lib/api'
import { getCallStateColor, formatRelativeTime } from '@/lib/utils'
import type { CallStatus } from '@/types'

export function ActiveCallsTable() {
  const { data: activeCalls, isLoading } = useQuery({
    queryKey: ['active-calls'],
    queryFn: async () => {
      // Fetch calls with status that indicates they're active
      const response = await api.getRecentCalls(50)
      // Filter to only active calls (in progress states)
      const activeStatuses = ['initiated', 'pending_review']
      return response.data?.filter(call =>
        call.status && activeStatuses.includes(call.status)
      ) || []
    },
    refetchInterval: 5000, // Refresh every 5 seconds for active calls
  })

  const formatCallState = (state: string) => {
    return state.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
  }

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-base font-semibold">Active Calls</CardTitle>
        <Phone className="h-4 w-4 text-muted-foreground" />
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        ) : activeCalls && activeCalls.length > 0 ? (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Provider</TableHead>
                <TableHead>Insurance</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Started</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {activeCalls.map((call) => (
                <TableRow key={call.id}>
                  <TableCell className="font-medium">
                    {call.provider_name}
                  </TableCell>
                  <TableCell>{call.insurance_name}</TableCell>
                  <TableCell>
                    <Badge className={getCallStateColor(call.status || 'initiated')}>
                      {formatCallState(call.status || 'initiated')}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-muted-foreground">
                    {call.created_at ? formatRelativeTime(call.created_at) : 'Just now'}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        ) : (
          <div className="flex flex-col items-center justify-center py-8 text-center">
            <Phone className="h-8 w-8 text-muted-foreground mb-2" />
            <p className="text-sm text-muted-foreground">No active calls</p>
            <p className="text-xs text-muted-foreground mt-1">
              Start a new call to see it here
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
