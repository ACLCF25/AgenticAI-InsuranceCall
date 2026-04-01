'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Phone, Plus, Trash2, PhoneCall, PhoneOff } from 'lucide-react'
import { toast } from 'sonner'
import { api } from '@/lib/api'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import type { TwilioNumber } from '@/types'

export default function PhoneNumbersPage() {
  const queryClient = useQueryClient()
  const [dialogOpen, setDialogOpen] = useState(false)
  const [newNumber, setNewNumber] = useState({ phone_number: '', friendly_name: '' })

  const { data, isLoading } = useQuery({
    queryKey: ['twilio-numbers'],
    queryFn: () => api.getTwilioNumbers(),
    refetchInterval: 10000,
  })

  const addMutation = useMutation({
    mutationFn: (num: typeof newNumber) => api.addTwilioNumber(num),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['twilio-numbers'] })
      toast.success('Phone number added')
      setDialogOpen(false)
      setNewNumber({ phone_number: '', friendly_name: '' })
    },
    onError: (err: any) => {
      toast.error(err.response?.data?.error || 'Failed to add number. Check the format is E.164 (e.g. +13513007215).')
    },
  })

  const toggleMutation = useMutation({
    mutationFn: ({ id, is_active }: { id: string; is_active: boolean }) =>
      api.updateTwilioNumber(id, { is_active }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['twilio-numbers'] })
      toast.success('Number updated')
    },
    onError: (err: any) => {
      toast.error(err.response?.data?.error || 'Failed to toggle number status. Please try again.')
    },
  })

  const deleteMutation = useMutation({
    mutationFn: (id: string) => api.deleteTwilioNumber(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['twilio-numbers'] })
      toast.success('Number removed')
    },
    onError: (err: any) => {
      toast.error(err.response?.data?.error || 'Failed to remove number. It may be in use by an active call.')
    },
  })

  const numbers: TwilioNumber[] = data?.numbers || []
  const available = numbers.filter((n) => n.is_active && !n.current_call_id).length
  const inUse = numbers.filter((n) => n.current_call_id).length
  const total = numbers.length

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="max-w-4xl space-y-6"
    >
      {/* Summary cards */}
      <div className="grid grid-cols-3 gap-4">
        <Card>
          <CardContent className="pt-6">
            <div className="text-2xl font-bold">{total}</div>
            <p className="text-xs text-muted-foreground">Total Lines</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="text-2xl font-bold text-green-600">{available}</div>
            <p className="text-xs text-muted-foreground">Available</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="text-2xl font-bold text-orange-600">{inUse}</div>
            <p className="text-xs text-muted-foreground">In Use</p>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Phone className="h-4 w-4 text-muted-foreground" />
              <div>
                <CardTitle className="text-base">Phone Lines</CardTitle>
                <CardDescription>Manage Twilio phone numbers for outbound calls</CardDescription>
              </div>
            </div>
            <Button size="sm" onClick={() => setDialogOpen(true)}>
              <Plus className="h-4 w-4 mr-1" />
              Add Number
            </Button>
            <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Add Phone Number</DialogTitle>
                  <DialogDescription>
                    Add a Twilio phone number to the calling pool. Must be in E.164 format.
                  </DialogDescription>
                </DialogHeader>
                <div className="space-y-4 py-4">
                  <div className="space-y-2">
                    <Label htmlFor="phone_number">Phone Number</Label>
                    <Input
                      id="phone_number"
                      value={newNumber.phone_number}
                      onChange={(e) => setNewNumber({ ...newNumber, phone_number: e.target.value })}
                      placeholder="+13513007215"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="friendly_name">Friendly Name (optional)</Label>
                    <Input
                      id="friendly_name"
                      value={newNumber.friendly_name}
                      onChange={(e) => setNewNumber({ ...newNumber, friendly_name: e.target.value })}
                      placeholder="Main Line"
                    />
                  </div>
                </div>
                <DialogFooter>
                  <Button variant="outline" onClick={() => setDialogOpen(false)}>Cancel</Button>
                  <Button
                    onClick={() => addMutation.mutate(newNumber)}
                    disabled={addMutation.isPending || !newNumber.phone_number}
                  >
                    {addMutation.isPending ? 'Adding...' : 'Add Number'}
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>
          </div>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="text-sm text-muted-foreground py-8 text-center">Loading numbers...</div>
          ) : numbers.length === 0 ? (
            <div className="text-sm text-muted-foreground py-8 text-center">
              No phone numbers configured. Add a Twilio number to start making calls.
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Phone Number</TableHead>
                  <TableHead>Name</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Active</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {numbers.map((num) => (
                  <TableRow key={num.id}>
                    <TableCell className="font-mono text-sm">{num.phone_number}</TableCell>
                    <TableCell className="text-muted-foreground">
                      {num.friendly_name || '-'}
                    </TableCell>
                    <TableCell>
                      {num.current_call_id ? (
                        <Badge className="bg-orange-500/10 text-orange-600 border-orange-500/20">
                          <PhoneCall className="h-3 w-3 mr-1" />
                          In Use
                        </Badge>
                      ) : num.is_active ? (
                        <Badge className="bg-green-500/10 text-green-600 border-green-500/20">
                          Available
                        </Badge>
                      ) : (
                        <Badge variant="outline">
                          <PhoneOff className="h-3 w-3 mr-1" />
                          Disabled
                        </Badge>
                      )}
                    </TableCell>
                    <TableCell>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => toggleMutation.mutate({ id: num.id, is_active: !num.is_active })}
                        disabled={toggleMutation.isPending}
                      >
                        {num.is_active ? 'Disable' : 'Enable'}
                      </Button>
                    </TableCell>
                    <TableCell className="text-right">
                      <Button
                        size="sm"
                        variant="ghost"
                        className="text-destructive hover:text-destructive"
                        onClick={() => deleteMutation.mutate(num.id)}
                        disabled={deleteMutation.isPending || !!num.current_call_id}
                        title={num.current_call_id ? 'Cannot delete while in use' : 'Delete number'}
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </motion.div>
  )
}
