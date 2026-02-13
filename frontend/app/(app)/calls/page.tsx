'use client'

import { useQuery } from '@tanstack/react-query'
import { useRouter } from 'next/navigation'
import { motion } from 'framer-motion'
import { Phone, Loader2, Search, ExternalLink } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { api } from '@/lib/api'
import { formatStatus, getStatusColor, formatRelativeTime, formatPhoneNumber } from '@/lib/utils'
import { useState } from 'react'

export default function CallsPage() {
  const router = useRouter()
  const [search, setSearch] = useState('')

  const { data, isLoading } = useQuery({
    queryKey: ['recent-calls', 100],
    queryFn: () => api.getRecentCalls(100),
    refetchInterval: 30000,
  })

  const calls = data?.data || []
  const filtered = calls.filter(
    (call) =>
      call.provider_name.toLowerCase().includes(search.toLowerCase()) ||
      call.insurance_name.toLowerCase().includes(search.toLowerCase()) ||
      call.npi.includes(search)
  )

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="text-base font-semibold">Call History</CardTitle>
            <Phone className="h-4 w-4 text-muted-foreground" />
          </div>
          <div className="relative mt-3">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search by provider, insurance, or NPI..."
              className="pl-9"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            </div>
          ) : filtered.length > 0 ? (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Provider</TableHead>
                  <TableHead>Insurance</TableHead>
                  <TableHead>NPI</TableHead>
                  <TableHead>Phone</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Date</TableHead>
                  <TableHead className="w-10" />
                </TableRow>
              </TableHeader>
              <TableBody>
                {filtered.map((call) => (
                  <TableRow
                    key={call.id}
                    className="cursor-pointer"
                    onClick={() => call.id && router.push(`/calls/${call.id}`)}
                  >
                    <TableCell className="font-medium">{call.provider_name}</TableCell>
                    <TableCell>{call.insurance_name}</TableCell>
                    <TableCell className="font-mono text-xs">{call.npi}</TableCell>
                    <TableCell className="text-xs">{formatPhoneNumber(call.insurance_phone)}</TableCell>
                    <TableCell>
                      <Badge className={getStatusColor(call.status || 'initiated')}>
                        {formatStatus(call.status || 'initiated')}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-muted-foreground text-xs">
                      {call.created_at ? formatRelativeTime(call.created_at) : '-'}
                    </TableCell>
                    <TableCell>
                      <ExternalLink className="h-3.5 w-3.5 text-muted-foreground" />
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <div className="flex flex-col items-center justify-center py-12 text-center">
              <Phone className="h-8 w-8 text-muted-foreground mb-2" />
              <p className="text-sm text-muted-foreground">
                {search ? 'No calls match your search' : 'No calls yet'}
              </p>
            </div>
          )}
        </CardContent>
      </Card>
    </motion.div>
  )
}
