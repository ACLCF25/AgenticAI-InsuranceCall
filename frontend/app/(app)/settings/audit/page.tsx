'use client'

import { motion } from 'framer-motion'
import { useQuery } from '@tanstack/react-query'
import { FileSearch, ShieldAlert } from 'lucide-react'
import { api } from '@/lib/api'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'

export default function AuditPage() {
  const { data, isLoading } = useQuery({
    queryKey: ['audit-logs'],
    queryFn: () => api.getAuditLogs(),
    refetchInterval: 30000,
  })

  const logs = data?.data || []

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <FileSearch className="h-4 w-4 text-muted-foreground" />
            <div>
              <CardTitle className="text-base">Audit Trail</CardTitle>
              <CardDescription>Recent system and account activity across the platform.</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="py-8 text-center text-sm text-muted-foreground">Loading audit events...</div>
          ) : logs.length === 0 ? (
            <div className="py-8 text-center text-sm text-muted-foreground">No audit events found.</div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>When</TableHead>
                  <TableHead>Action</TableHead>
                  <TableHead>Resource</TableHead>
                  <TableHead>User ID</TableHead>
                  <TableHead>Details</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {logs.map((log) => (
                  <TableRow key={log.id}>
                    <TableCell className="text-xs text-muted-foreground">
                      {new Date(log.timestamp).toLocaleString()}
                    </TableCell>
                    <TableCell className="font-medium">{log.action}</TableCell>
                    <TableCell className="text-xs text-muted-foreground">
                      {[log.resource_type, log.resource_id].filter(Boolean).join(' / ') || 'System'}
                    </TableCell>
                    <TableCell className="font-mono text-xs">{log.user_id || '-'}</TableCell>
                    <TableCell className="max-w-[360px] text-xs text-muted-foreground">
                      <pre className="whitespace-pre-wrap break-words font-mono">
                        {log.details ? JSON.stringify(log.details, null, 2) : '{}'}
                      </pre>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardContent className="flex items-center gap-3 pt-6 text-sm text-muted-foreground">
          <ShieldAlert className="h-4 w-4" />
          Audit data is restricted to super admins because it includes privileged account and system activity.
        </CardContent>
      </Card>
    </motion.div>
  )
}
