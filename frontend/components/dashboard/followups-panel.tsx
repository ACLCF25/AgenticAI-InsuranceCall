// components/dashboard/followups-panel.tsx
'use client'

import { Calendar, Clock, Phone, FileText, Play, Loader2 } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { api } from '@/lib/api'
import { formatRelativeTime } from '@/lib/utils'
import type { ScheduledFollowup } from '@/types'
import { useMutation, useQueryClient } from '@tanstack/react-query'

interface FollowupsPanelProps {
  followups: ScheduledFollowup[]
  showAll?: boolean
}

export function FollowupsPanel({ followups, showAll = false }: FollowupsPanelProps) {
  const queryClient = useQueryClient()

  const displayFollowups = showAll ? followups : followups.slice(0, 5)

  const executeFollowup = useMutation({
    mutationFn: (followupId: string) => api.executeFollowup(followupId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['followups'] })
    },
  })

  const getActionIcon = (actionType: string) => {
    switch (actionType) {
      case 'retry_call':
        return <Phone className="h-4 w-4" />
      case 'follow_up_call':
        return <Clock className="h-4 w-4" />
      case 'submit_documents_then_call':
        return <FileText className="h-4 w-4" />
      default:
        return <Calendar className="h-4 w-4" />
    }
  }

  const getActionLabel = (actionType: string) => {
    switch (actionType) {
      case 'retry_call':
        return 'Retry Call'
      case 'follow_up_call':
        return 'Follow-up'
      case 'submit_documents_then_call':
        return 'Submit Docs'
      default:
        return actionType
    }
  }

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'pending':
        return <Badge className="border-amber-500/25 bg-amber-500/10 text-amber-700 dark:text-amber-300">Pending</Badge>
      case 'completed':
        return <Badge className="border-emerald-500/25 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300">Completed</Badge>
      case 'failed':
        return <Badge className="border-red-500/25 bg-red-500/10 text-red-700 dark:text-red-300">Failed</Badge>
      default:
        return <Badge>{status}</Badge>
    }
  }

  const isOverdue = (scheduledDate: string) => {
    return new Date(scheduledDate) < new Date()
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-base font-semibold tracking-tight">
              Scheduled Follow-ups
            </CardTitle>
            <CardDescription>
              {followups.length} pending action{followups.length !== 1 ? 's' : ''}
            </CardDescription>
          </div>
          <Calendar className="h-4 w-4 text-muted-foreground" />
        </div>
      </CardHeader>
      <CardContent>
        {displayFollowups.length > 0 ? (
          <div className="space-y-4">
            {displayFollowups.map((followup) => (
              <div
                key={followup.id}
                className="flex items-start justify-between rounded-xl border border-border/60 bg-background/50 p-3"
              >
                <div className="flex items-start gap-3">
                  <div className="mt-0.5 rounded-lg bg-muted p-2 text-muted-foreground">
                    {getActionIcon(followup.action_type)}
                  </div>
                  <div className="space-y-1">
                    <p className="text-sm font-medium">
                      {followup.provider_name}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {followup.insurance_name}
                    </p>
                    <div className="mt-2 flex items-center gap-2">
                      <Badge variant="outline" className="text-xs">
                        {getActionLabel(followup.action_type)}
                      </Badge>
                      {getStatusBadge(followup.status)}
                    </div>
                    <p className={`mt-1 text-xs ${
                      isOverdue(followup.scheduled_date)
                        ? 'font-medium text-red-600 dark:text-red-300'
                        : 'text-muted-foreground'
                    }`}>
                      {isOverdue(followup.scheduled_date) ? 'Overdue: ' : 'Due: '}
                      {formatRelativeTime(followup.scheduled_date)}
                    </p>
                  </div>
                </div>
                {followup.status === 'pending' && (
                  <Button
                    size="sm"
                    variant="outline"
                    className="rounded-full"
                    onClick={() => executeFollowup.mutate(followup.id)}
                    disabled={executeFollowup.isPending}
                  >
                    {executeFollowup.isPending ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <>
                        <Play className="mr-1 h-3 w-3" />
                        Execute
                      </>
                    )}
                  </Button>
                )}
              </div>
            ))}

            {!showAll && followups.length > 5 && (
              <Button variant="ghost" className="w-full rounded-xl" size="sm">
                View all {followups.length} follow-ups
              </Button>
            )}
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center py-10 text-center">
            <Calendar className="mb-2 h-8 w-8 text-muted-foreground" />
            <p className="text-sm text-muted-foreground">
              No scheduled follow-ups
            </p>
            <p className="mt-1 text-xs text-muted-foreground">
              Follow-ups will appear here when calls require action
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
