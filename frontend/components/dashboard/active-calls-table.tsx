'use client'

import { useQuery } from '@tanstack/react-query'
import { useRouter } from 'next/navigation'
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
import { getCallStateColor, formatStatus, formatRelativeTime } from '@/lib/utils'

export function ActiveCallsTable() {
  const router = useRouter()

  const { data: activeCalls, isLoading } = useQuery({
    queryKey: ['active-calls'],
    queryFn: async () => {
      const response = await api.getRecentCalls(50)
      const activeStatuses = ['initiated', 'pending_review']
      return response.data?.filter(call =>
        call.status && activeStatuses.includes(call.status)
      ) || []
    },
    refetchInterval: 5000,
  })

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3">
        <CardTitle className="text-base font-semibold tracking-tight">Active Calls</CardTitle>
        {activeCalls && activeCalls.length > 0 && (
          <div className="flex items-center gap-2">
            <div className="pulse-dot h-2 w-2 rounded-full bg-primary" />
            <span className="text-xs text-muted-foreground">{activeCalls.length} active</span>
          </div>
        )}
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
                <TableRow
                  key={call.id}
                  className="cursor-pointer"
                  onClick={() => call.id && router.push(`/calls/${call.id}`)}
                >
                  <TableCell className="font-medium">
                    {call.provider_name}
                  </TableCell>
                  <TableCell>{call.insurance_name}</TableCell>
                  <TableCell>
                    <Badge className={getCallStateColor(call.status || 'initiated')}>
                      {formatStatus(call.status || 'initiated')}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-muted-foreground text-xs">
                    {call.created_at ? formatRelativeTime(call.created_at) : 'Just now'}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        ) : (
          <div className="flex flex-col items-center justify-center py-10 text-center">
            <Phone className="mb-2 h-8 w-8 text-muted-foreground" />
            <p className="text-sm text-muted-foreground">No active calls</p>
            <p className="mt-1 text-xs text-muted-foreground">
              Start a new call to see it here
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
