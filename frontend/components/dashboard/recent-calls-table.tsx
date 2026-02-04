// components/dashboard/recent-calls-table.tsx
'use client'

import { useQuery } from '@tanstack/react-query'
import { useRouter } from 'next/navigation'
import { Phone, Loader2, ExternalLink, Eye } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { api } from '@/lib/api'
import { getStatusColor, formatDate, formatPhoneNumber } from '@/lib/utils'

interface RecentCallsTableProps {
  showAll?: boolean
}

export function RecentCallsTable({ showAll = false }: RecentCallsTableProps) {
  const router = useRouter()
  const limit = showAll ? 100 : 10

  const { data: calls, isLoading } = useQuery({
    queryKey: ['recent-calls', limit],
    queryFn: () => api.getRecentCalls(limit),
    refetchInterval: 30000, // Refresh every 30 seconds
  })

  const formatStatus = (status: string) => {
    return status.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Recent Calls</CardTitle>
            <CardDescription>
              {showAll ? 'All credentialing calls' : 'Latest credentialing calls'}
            </CardDescription>
          </div>
          {!showAll && calls?.data && calls.data.length > 0 && (
            <Button variant="outline" size="sm">
              View All
              <ExternalLink className="ml-2 h-3 w-3" />
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        ) : calls?.data && calls.data.length > 0 ? (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Provider</TableHead>
                <TableHead>Insurance</TableHead>
                <TableHead>NPI</TableHead>
                <TableHead>Phone</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Reference #</TableHead>
                <TableHead>Date</TableHead>
                <TableHead></TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {calls.data.map((call) => (
                <TableRow
                  key={call.id}
                  className="cursor-pointer hover:bg-muted/50"
                  onClick={() => router.push(`/calls/${call.id}`)}
                >
                  <TableCell className="font-medium">
                    {call.provider_name}
                  </TableCell>
                  <TableCell>{call.insurance_name}</TableCell>
                  <TableCell className="font-mono text-sm">
                    {call.npi}
                  </TableCell>
                  <TableCell className="text-sm">
                    {formatPhoneNumber(call.insurance_phone)}
                  </TableCell>
                  <TableCell>
                    <Badge className={getStatusColor(call.status || 'initiated')}>
                      {formatStatus(call.status || 'initiated')}
                    </Badge>
                  </TableCell>
                  <TableCell className="font-mono text-sm">
                    {call.reference_number || '-'}
                  </TableCell>
                  <TableCell className="text-muted-foreground text-sm">
                    {call.created_at ? formatDate(call.created_at) : '-'}
                  </TableCell>
                  <TableCell>
                    <Button variant="ghost" size="sm">
                      <Eye className="h-4 w-4" />
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        ) : (
          <div className="flex flex-col items-center justify-center py-8 text-center">
            <Phone className="h-8 w-8 text-muted-foreground mb-2" />
            <p className="text-sm text-muted-foreground">No calls yet</p>
            <p className="text-xs text-muted-foreground mt-1">
              Start your first credentialing call
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
