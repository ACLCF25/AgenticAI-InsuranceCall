// components/dashboard/followups-panel.tsx
'use client'

import { Calendar, Clock, Phone, FileText, Play, Loader2 } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { api } from '@/lib/api'
import { formatDate, formatRelativeTime } from '@/lib/utils'
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
        return <Badge className="bg-yellow-100 text-yellow-800">Pending</Badge>
      case 'completed':
        return <Badge className="bg-green-100 text-green-800">Completed</Badge>
      case 'failed':
        return <Badge className="bg-red-100 text-red-800">Failed</Badge>
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
            <CardTitle className="text-base font-semibold">
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
                className="flex items-start justify-between p-3 rounded-lg border bg-card"
              >
                <div className="flex items-start gap-3">
                  <div className="mt-0.5 p-2 rounded-full bg-muted">
                    {getActionIcon(followup.action_type)}
                  </div>
                  <div className="space-y-1">
                    <p className="font-medium text-sm">
                      {followup.provider_name}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {followup.insurance_name}
                    </p>
                    <div className="flex items-center gap-2 mt-2">
                      <Badge variant="outline" className="text-xs">
                        {getActionLabel(followup.action_type)}
                      </Badge>
                      {getStatusBadge(followup.status)}
                    </div>
                    <p className={`text-xs mt-1 ${
                      isOverdue(followup.scheduled_date)
                        ? 'text-red-600 font-medium'
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
                    onClick={() => executeFollowup.mutate(followup.id)}
                    disabled={executeFollowup.isPending}
                  >
                    {executeFollowup.isPending ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <>
                        <Play className="h-3 w-3 mr-1" />
                        Execute
                      </>
                    )}
                  </Button>
                )}
              </div>
            ))}

            {!showAll && followups.length > 5 && (
              <Button variant="ghost" className="w-full" size="sm">
                View all {followups.length} follow-ups
              </Button>
            )}
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center py-8 text-center">
            <Calendar className="h-8 w-8 text-muted-foreground mb-2" />
            <p className="text-sm text-muted-foreground">
              No scheduled follow-ups
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              Follow-ups will appear here when calls require action
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
