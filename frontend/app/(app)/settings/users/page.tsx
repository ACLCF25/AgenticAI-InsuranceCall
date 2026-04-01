'use client'

import { motion } from 'framer-motion'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Shield, ShieldCheck, UserCheck, UserX, MailCheck, MailX, Clock3 } from 'lucide-react'
import { toast } from 'sonner'
import { api } from '@/lib/api'
import { useAuth } from '@/lib/auth-context'
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
import type { AdminUser, UserRole } from '@/types'

const roleOptions: UserRole[] = ['agent', 'admin', 'super_admin']

export default function UsersPage() {
  const queryClient = useQueryClient()
  const { user: currentUser } = useAuth()

  const { data, isLoading } = useQuery({
    queryKey: ['admin-users'],
    queryFn: () => api.getUsers(),
    refetchInterval: 15000,
  })

  const updateMutation = useMutation({
    mutationFn: ({ userId, updates }: { userId: string; updates: { approval_status?: string; role?: string } }) =>
      api.updateUser(userId, updates),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['admin-users'] })
      toast.success('User updated')
    },
    onError: (err: any) => {
      toast.error(err.response?.data?.error || 'Failed to update user')
    },
  })

  const users: AdminUser[] = data?.users || []

  const getRoleBadge = (role: UserRole | null) => {
    if (!role) {
      return <Badge variant="outline">Unknown</Badge>
    }
    if (role === 'super_admin') {
      return <Badge className="bg-amber-500/10 text-amber-700 border-amber-500/20">Super Admin</Badge>
    }
    if (role === 'admin') {
      return <Badge className="bg-blue-500/10 text-blue-700 border-blue-500/20">Admin</Badge>
    }
    return <Badge variant="outline">Agent</Badge>
  }

  const getApprovalBadge = (user: AdminUser) => {
    if (!user.email_confirmed) {
      return <Badge className="bg-slate-500/10 text-slate-700 border-slate-500/20">Awaiting Email</Badge>
    }
    if (user.approval_status === 'approved') {
      return <Badge className="bg-green-500/10 text-green-700 border-green-500/20">Approved</Badge>
    }
    if (user.approval_status === 'rejected') {
      return <Badge variant="destructive">Rejected</Badge>
    }
    return <Badge className="bg-yellow-500/10 text-yellow-700 border-yellow-500/20">Pending Approval</Badge>
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="max-w-5xl space-y-6"
    >
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <ShieldCheck className="h-4 w-4 text-muted-foreground" />
            <div>
              <CardTitle className="text-base">User Access</CardTitle>
              <CardDescription>
                {currentUser?.role === 'super_admin'
                  ? 'Approve accounts, review verification state, and manage roles.'
                  : 'Approve or reject agent accounts after email verification.'}
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="py-8 text-center text-sm text-muted-foreground">Loading users...</div>
          ) : users.length === 0 ? (
            <div className="py-8 text-center text-sm text-muted-foreground">No users found</div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Username</TableHead>
                  <TableHead>Email</TableHead>
                  <TableHead>Verification</TableHead>
                  <TableHead>Approval</TableHead>
                  <TableHead>Role</TableHead>
                  <TableHead>Created</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {users.map((user) => (
                  <TableRow key={user.id}>
                    <TableCell className="font-medium">{user.username || 'Pending username'}</TableCell>
                    <TableCell className="text-muted-foreground">{user.email}</TableCell>
                    <TableCell>
                      {user.email_confirmed ? (
                        <div className="flex items-center gap-1 text-xs text-green-700">
                          <MailCheck className="h-3.5 w-3.5" />
                          Verified
                        </div>
                      ) : (
                        <div className="flex items-center gap-1 text-xs text-muted-foreground">
                          <MailX className="h-3.5 w-3.5" />
                          Waiting
                        </div>
                      )}
                    </TableCell>
                    <TableCell>{getApprovalBadge(user)}</TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        {getRoleBadge(user.role)}
                        {currentUser?.role === 'super_admin' && (
                          <select
                            className="h-8 rounded-md border border-input bg-transparent px-2 text-xs"
                            value={user.role || 'agent'}
                            onChange={(e) =>
                              updateMutation.mutate({
                                userId: user.id,
                                updates: { role: e.target.value },
                              })
                            }
                            disabled={updateMutation.isPending}
                          >
                            {roleOptions.map((role) => (
                              <option key={role} value={role}>
                                {role.replace('_', ' ')}
                              </option>
                            ))}
                          </select>
                        )}
                      </div>
                    </TableCell>
                    <TableCell className="text-xs text-muted-foreground">
                      {user.created_at ? new Date(user.created_at).toLocaleDateString() : 'Unknown'}
                    </TableCell>
                    <TableCell className="text-right">
                      <div className="flex justify-end gap-2">
                        {user.approval_status !== 'approved' && (
                          <Button
                            size="sm"
                            variant="outline"
                            className="text-green-700 border-green-600/30 hover:bg-green-500/10"
                            onClick={() =>
                              updateMutation.mutate({
                                userId: user.id,
                                updates: { approval_status: 'approved' },
                              })
                            }
                            disabled={updateMutation.isPending || !user.email_confirmed}
                            title={!user.email_confirmed ? 'Email confirmation is required first' : 'Approve account'}
                          >
                            <UserCheck className="mr-1 h-3.5 w-3.5" />
                            Approve
                          </Button>
                        )}
                        {user.approval_status !== 'rejected' && (
                          <Button
                            size="sm"
                            variant="outline"
                            className="text-destructive border-destructive/30 hover:bg-destructive/10"
                            onClick={() =>
                              updateMutation.mutate({
                                userId: user.id,
                                updates: { approval_status: 'rejected' },
                              })
                            }
                            disabled={updateMutation.isPending}
                          >
                            <UserX className="mr-1 h-3.5 w-3.5" />
                            Reject
                          </Button>
                        )}
                        {user.approval_status === 'approved' && (
                          <div className="flex items-center gap-1 text-xs text-muted-foreground">
                            <Clock3 className="h-3.5 w-3.5" />
                            Live
                          </div>
                        )}
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardContent className="pt-6 text-sm text-muted-foreground">
          {currentUser?.role === 'super_admin' ? (
            <div className="space-y-2">
              <p className="flex items-center gap-2">
                <Shield className="h-4 w-4" />
                Super admins can approve all users and change any role.
              </p>
              <p>Admins can only review and approve agent accounts.</p>
            </div>
          ) : (
            <p>Only email-confirmed agent accounts can be approved from this screen.</p>
          )}
        </CardContent>
      </Card>
    </motion.div>
  )
}
