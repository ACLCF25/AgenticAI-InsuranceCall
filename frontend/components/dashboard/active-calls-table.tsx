'use client'

import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { useRouter } from 'next/navigation'
import { Phone, Loader2, UserCheck } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogFooter,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog'
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
import { toast } from 'sonner'

interface TakeOverDialogState {
  open: boolean
  callId: string
  agentPhone: string
}

export function ActiveCallsTable() {
  const router = useRouter()
  const [takeOverDialog, setTakeOverDialog] = useState<TakeOverDialogState>({
    open: false,
    callId: '',
    agentPhone: '',
  })
  // Track which call IDs have been successfully handed off to a human agent
  const [agentConnectedIds, setAgentConnectedIds] = useState<Set<string>>(new Set())

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

  const transferMutation = useMutation({
    mutationFn: ({ callId, agentPhone }: { callId: string; agentPhone: string }) =>
      api.transferToAgent(callId, agentPhone),
    onSuccess: (_, variables) => {
      setAgentConnectedIds((prev) => new Set([...prev, variables.callId]))
      setTakeOverDialog({ open: false, callId: '', agentPhone: '' })
      toast.success('Agent connected', {
        description: `Call transferred to ${variables.agentPhone}`,
      })
    },
    onError: (error: any) => {
      toast.error('Transfer failed', {
        description: error.response?.data?.error || error.message,
      })
    },
  })

  const openTakeOverDialog = (callId: string, e: React.MouseEvent) => {
    e.stopPropagation()
    setTakeOverDialog({ open: true, callId, agentPhone: '' })
  }

  const confirmTakeOver = () => {
    if (!takeOverDialog.agentPhone.trim()) return
    transferMutation.mutate({
      callId: takeOverDialog.callId,
      agentPhone: takeOverDialog.agentPhone,
    })
  }

  return (
    <>
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
                  <TableHead />
                </TableRow>
              </TableHeader>
              <TableBody>
                {activeCalls.map((call) => {
                  const isAgentConnected = call.id ? agentConnectedIds.has(call.id) : false
                  // Show Take Over only for AI-mode calls (call_mode === 'ai' or call_mode absent)
                  const isAiMode = !call.call_mode || call.call_mode === 'ai'

                  return (
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
                        {isAgentConnected ? (
                          <Badge className="bg-green-500/15 text-green-600 hover:bg-green-500/20">
                            Agent Connected
                          </Badge>
                        ) : (
                          <Badge className={getCallStateColor(call.status || 'initiated')}>
                            {formatStatus(call.status || 'initiated')}
                          </Badge>
                        )}
                      </TableCell>
                      <TableCell className="text-muted-foreground text-xs">
                        {call.created_at ? formatRelativeTime(call.created_at) : 'Just now'}
                      </TableCell>
                      <TableCell className="text-right">
                        {isAiMode && !isAgentConnected && call.id && (
                          <Button
                            size="sm"
                            variant="outline"
                            className="h-7 gap-1.5 text-xs"
                            onClick={(e) => openTakeOverDialog(call.id!, e)}
                          >
                            <UserCheck className="h-3.5 w-3.5" />
                            Take Over
                          </Button>
                        )}
                      </TableCell>
                    </TableRow>
                  )
                })}
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

      <Dialog
        open={takeOverDialog.open}
        onOpenChange={(open) => {
          if (!open) setTakeOverDialog({ open: false, callId: '', agentPhone: '' })
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Take Over Call</DialogTitle>
            <DialogDescription>
              Enter the real agent&apos;s phone number to transfer this call. The AI agent will
              hand off to the specified number.
            </DialogDescription>
          </DialogHeader>
          <div className="py-2">
            <label className="mb-1.5 block text-sm font-medium">Agent Phone Number</label>
            <Input
              placeholder="+1 (555) 000-0000"
              value={takeOverDialog.agentPhone}
              onChange={(e) =>
                setTakeOverDialog((prev) => ({ ...prev, agentPhone: e.target.value }))
              }
              onKeyDown={(e) => {
                if (e.key === 'Enter') confirmTakeOver()
              }}
              autoFocus
            />
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setTakeOverDialog({ open: false, callId: '', agentPhone: '' })}
              disabled={transferMutation.isPending}
            >
              Cancel
            </Button>
            <Button
              onClick={confirmTakeOver}
              disabled={!takeOverDialog.agentPhone.trim() || transferMutation.isPending}
            >
              {transferMutation.isPending ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <UserCheck className="mr-2 h-4 w-4" />
              )}
              Transfer Call
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}
