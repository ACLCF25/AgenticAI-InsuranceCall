'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Users, Plus, Shield, ShieldCheck, UserX, UserCheck } from 'lucide-react'
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
import type { AdminUser } from '@/types'

export default function UsersPage() {
  const queryClient = useQueryClient()
  const [dialogOpen, setDialogOpen] = useState(false)
  const [newUser, setNewUser] = useState({ username: '', email: '', password: '', role: 'user' })

  const { data, isLoading } = useQuery({
    queryKey: ['admin-users'],
    queryFn: () => api.getUsers(),
    refetchInterval: 10000,
  })

  const createMutation = useMutation({
    mutationFn: (user: typeof newUser) => api.createUser(user),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['admin-users'] })
      toast.success('User created (pending approval)')
      setDialogOpen(false)
      setNewUser({ username: '', email: '', password: '', role: 'user' })
    },
    onError: (err: any) => {
      toast.error(err.response?.data?.error || 'Failed to create user')
    },
  })

  const updateMutation = useMutation({
    mutationFn: ({ userId, updates }: { userId: string; updates: { is_active?: boolean; role?: string } }) =>
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

  const getStatusBadge = (user: AdminUser) => {
    if (user.is_active) {
      return <Badge className="bg-green-500/10 text-green-600 border-green-500/20">Active</Badge>
    }
    if (user.last_login === null) {
      return <Badge className="bg-yellow-500/10 text-yellow-600 border-yellow-500/20">Pending Approval</Badge>
    }
    return <Badge variant="destructive">Deactivated</Badge>
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="max-w-4xl space-y-6"
    >
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Users className="h-4 w-4 text-muted-foreground" />
              <div>
                <CardTitle className="text-base">User Management</CardTitle>
                <CardDescription>Manage user accounts and approvals</CardDescription>
              </div>
            </div>
            <Button size="sm" onClick={() => setDialogOpen(true)}>
              <Plus className="h-4 w-4 mr-1" />
              Add User
            </Button>
            <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Add New User</DialogTitle>
                  <DialogDescription>
                    Create a new user account. The account will require admin approval before the user can log in.
                  </DialogDescription>
                </DialogHeader>
                <div className="space-y-4 py-4">
                  <div className="space-y-2">
                    <Label htmlFor="username">Username</Label>
                    <Input
                      id="username"
                      value={newUser.username}
                      onChange={(e) => setNewUser({ ...newUser, username: e.target.value })}
                      placeholder="johndoe"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="email">Email</Label>
                    <Input
                      id="email"
                      type="email"
                      value={newUser.email}
                      onChange={(e) => setNewUser({ ...newUser, email: e.target.value })}
                      placeholder="john@example.com"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="password">Password</Label>
                    <Input
                      id="password"
                      type="password"
                      value={newUser.password}
                      onChange={(e) => setNewUser({ ...newUser, password: e.target.value })}
                      placeholder="Min 8 characters"
                    />
                    <p className="text-xs text-muted-foreground">Minimum 8 characters</p>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="role">Role</Label>
                    <select
                      id="role"
                      value={newUser.role}
                      onChange={(e) => setNewUser({ ...newUser, role: e.target.value })}
                      className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                    >
                      <option value="user">User</option>
                      <option value="admin">Admin</option>
                    </select>
                  </div>
                </div>
                <DialogFooter>
                  <Button variant="outline" onClick={() => setDialogOpen(false)}>Cancel</Button>
                  <Button
                    onClick={() => createMutation.mutate(newUser)}
                    disabled={createMutation.isPending || !newUser.username || !newUser.email || newUser.password.length < 8}
                  >
                    {createMutation.isPending ? 'Creating...' : 'Create User'}
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>
          </div>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="text-sm text-muted-foreground py-8 text-center">Loading users...</div>
          ) : users.length === 0 ? (
            <div className="text-sm text-muted-foreground py-8 text-center">No users found</div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Username</TableHead>
                  <TableHead>Email</TableHead>
                  <TableHead>Role</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Last Login</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {users.map((user) => (
                  <TableRow key={user.id}>
                    <TableCell className="font-medium">{user.username}</TableCell>
                    <TableCell className="text-muted-foreground">{user.email}</TableCell>
                    <TableCell>
                      {user.role === 'admin' ? (
                        <div className="flex items-center gap-1 text-xs">
                          <ShieldCheck className="h-3.5 w-3.5 text-primary" />
                          Admin
                        </div>
                      ) : (
                        <div className="flex items-center gap-1 text-xs">
                          <Shield className="h-3.5 w-3.5 text-muted-foreground" />
                          User
                        </div>
                      )}
                    </TableCell>
                    <TableCell>{getStatusBadge(user)}</TableCell>
                    <TableCell className="text-muted-foreground text-xs">
                      {user.last_login
                        ? new Date(user.last_login).toLocaleDateString()
                        : 'Never'}
                    </TableCell>
                    <TableCell className="text-right">
                      {!user.is_active && user.last_login === null ? (
                        <Button
                          size="sm"
                          variant="outline"
                          className="text-green-600 border-green-600/30 hover:bg-green-500/10"
                          onClick={() => updateMutation.mutate({ userId: user.id, updates: { is_active: true } })}
                          disabled={updateMutation.isPending}
                        >
                          <UserCheck className="h-3.5 w-3.5 mr-1" />
                          Approve
                        </Button>
                      ) : user.is_active ? (
                        <Button
                          size="sm"
                          variant="outline"
                          className="text-destructive border-destructive/30 hover:bg-destructive/10"
                          onClick={() => updateMutation.mutate({ userId: user.id, updates: { is_active: false } })}
                          disabled={updateMutation.isPending}
                        >
                          <UserX className="h-3.5 w-3.5 mr-1" />
                          Deactivate
                        </Button>
                      ) : (
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => updateMutation.mutate({ userId: user.id, updates: { is_active: true } })}
                          disabled={updateMutation.isPending}
                        >
                          <UserCheck className="h-3.5 w-3.5 mr-1" />
                          Reactivate
                        </Button>
                      )}
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
