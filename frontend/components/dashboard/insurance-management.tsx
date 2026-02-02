// components/dashboard/insurance-management.tsx
'use client'

import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Building2,
  Loader2,
  Plus,
  Trash2,
  Edit,
  ChevronDown,
  ChevronRight,
  Phone,
  Hash
} from 'lucide-react'
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
import { toast } from 'sonner'
import type { InsuranceProvider, IVRKnowledge } from '@/types'
import { AddInsuranceDialog } from './add-insurance-dialog'
import { AddIVRDialog } from './add-ivr-dialog'

export function InsuranceManagement() {
  const queryClient = useQueryClient()
  const [expandedRows, setExpandedRows] = useState<Set<string>>(new Set())
  const [addInsuranceOpen, setAddInsuranceOpen] = useState(false)
  const [editingProvider, setEditingProvider] = useState<InsuranceProvider | null>(null)
  const [addIVROpen, setAddIVROpen] = useState(false)
  const [selectedInsurance, setSelectedInsurance] = useState<string>('')

  // Fetch insurance providers
  const { data: providers, isLoading: loadingProviders } = useQuery({
    queryKey: ['insurance-providers'],
    queryFn: () => api.getInsuranceProviders(),
  })

  // Delete provider mutation
  const deleteProviderMutation = useMutation({
    mutationFn: (id: string) => api.deleteInsuranceProvider(id),
    onSuccess: () => {
      toast.success('Insurance provider deleted')
      queryClient.invalidateQueries({ queryKey: ['insurance-providers'] })
    },
    onError: (error: any) => {
      toast.error('Failed to delete provider', {
        description: error.response?.data?.error || error.message
      })
    }
  })

  // Delete IVR mutation
  const deleteIVRMutation = useMutation({
    mutationFn: (id: string) => api.deleteIVRKnowledge(id),
    onSuccess: () => {
      toast.success('IVR step deleted')
      queryClient.invalidateQueries({ queryKey: ['ivr-knowledge'] })
    },
    onError: (error: any) => {
      toast.error('Failed to delete IVR step', {
        description: error.response?.data?.error || error.message
      })
    }
  })

  const toggleRow = (id: string) => {
    const newExpanded = new Set(expandedRows)
    if (newExpanded.has(id)) {
      newExpanded.delete(id)
    } else {
      newExpanded.add(id)
    }
    setExpandedRows(newExpanded)
  }

  const handleAddIVR = (insuranceName: string) => {
    setSelectedInsurance(insuranceName)
    setAddIVROpen(true)
  }

  const handleEditProvider = (provider: InsuranceProvider) => {
    setEditingProvider(provider)
    setAddInsuranceOpen(true)
  }

  const getActionBadge = (action: string) => {
    switch (action) {
      case 'dtmf':
        return <Badge variant="secondary">Press Button</Badge>
      case 'speech':
        return <Badge variant="outline">Say Phrase</Badge>
      case 'wait':
        return <Badge>Wait</Badge>
      default:
        return <Badge variant="secondary">{action}</Badge>
    }
  }

  return (
    <>
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Insurance Providers</CardTitle>
              <CardDescription>
                Manage insurance companies and their IVR menu configurations
              </CardDescription>
            </div>
            <Button onClick={() => { setEditingProvider(null); setAddInsuranceOpen(true) }}>
              <Plus className="mr-2 h-4 w-4" />
              Add Insurance
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {loadingProviders ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            </div>
          ) : providers?.data && providers.data.length > 0 ? (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-8"></TableHead>
                  <TableHead>Insurance Name</TableHead>
                  <TableHead>Phone Number</TableHead>
                  <TableHead>Department</TableHead>
                  <TableHead>Avg Wait Time</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {providers.data.map((provider) => (
                  <InsuranceRow
                    key={provider.id}
                    provider={provider}
                    isExpanded={expandedRows.has(provider.id!)}
                    onToggle={() => toggleRow(provider.id!)}
                    onEdit={() => handleEditProvider(provider)}
                    onDelete={() => deleteProviderMutation.mutate(provider.id!)}
                    onAddIVR={() => handleAddIVR(provider.insurance_name)}
                    onDeleteIVR={(id) => deleteIVRMutation.mutate(id)}
                    getActionBadge={getActionBadge}
                  />
                ))}
              </TableBody>
            </Table>
          ) : (
            <div className="flex flex-col items-center justify-center py-8 text-center">
              <Building2 className="h-8 w-8 text-muted-foreground mb-2" />
              <p className="text-sm text-muted-foreground">No insurance providers yet</p>
              <p className="text-xs text-muted-foreground mt-1">
                Add your first insurance provider to configure IVR menus
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      <AddInsuranceDialog
        open={addInsuranceOpen}
        onOpenChange={setAddInsuranceOpen}
        editingProvider={editingProvider}
      />

      <AddIVRDialog
        open={addIVROpen}
        onOpenChange={setAddIVROpen}
        insuranceName={selectedInsurance}
      />
    </>
  )
}

interface InsuranceRowProps {
  provider: InsuranceProvider
  isExpanded: boolean
  onToggle: () => void
  onEdit: () => void
  onDelete: () => void
  onAddIVR: () => void
  onDeleteIVR: (id: string) => void
  getActionBadge: (action: string) => React.ReactNode
}

function InsuranceRow({
  provider,
  isExpanded,
  onToggle,
  onEdit,
  onDelete,
  onAddIVR,
  onDeleteIVR,
  getActionBadge
}: InsuranceRowProps) {
  const { data: ivrData, isLoading: loadingIVR } = useQuery({
    queryKey: ['ivr-knowledge', provider.insurance_name],
    queryFn: () => api.getIVRKnowledge(provider.insurance_name),
    enabled: isExpanded,
  })

  return (
    <>
      <TableRow className="cursor-pointer" onClick={onToggle}>
        <TableCell>
          {isExpanded ? (
            <ChevronDown className="h-4 w-4" />
          ) : (
            <ChevronRight className="h-4 w-4" />
          )}
        </TableCell>
        <TableCell className="font-medium">{provider.insurance_name}</TableCell>
        <TableCell>
          <div className="flex items-center gap-1">
            <Phone className="h-3 w-3 text-muted-foreground" />
            {provider.phone_number}
          </div>
        </TableCell>
        <TableCell>{provider.department || '-'}</TableCell>
        <TableCell>
          {provider.average_wait_time_minutes
            ? `${provider.average_wait_time_minutes} min`
            : '-'}
        </TableCell>
        <TableCell className="text-right">
          <div className="flex justify-end gap-2" onClick={(e) => e.stopPropagation()}>
            <Button variant="outline" size="sm" onClick={onEdit}>
              <Edit className="h-3 w-3" />
            </Button>
            <Button variant="outline" size="sm" onClick={onDelete}>
              <Trash2 className="h-3 w-3 text-destructive" />
            </Button>
          </div>
        </TableCell>
      </TableRow>

      {isExpanded && (
        <TableRow>
          <TableCell colSpan={6} className="bg-muted/30 p-4">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <h4 className="font-medium text-sm">IVR Navigation Sequence</h4>
                <Button variant="outline" size="sm" onClick={onAddIVR}>
                  <Plus className="mr-1 h-3 w-3" />
                  Add IVR Step
                </Button>
              </div>

              {loadingIVR ? (
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Loading IVR steps...
                </div>
              ) : ivrData?.data && ivrData.data.length > 0 ? (
                <div className="space-y-2">
                  {ivrData.data
                    .sort((a, b) => a.menu_level - b.menu_level)
                    .map((ivr, index) => (
                      <div
                        key={ivr.id}
                        className="flex items-center gap-3 p-3 bg-background rounded-lg border"
                      >
                        <div className="flex items-center justify-center w-6 h-6 rounded-full bg-primary/10 text-primary text-xs font-bold">
                          {ivr.menu_level}
                        </div>
                        <div className="flex-1">
                          <p className="text-sm font-medium">
                            Listen for: <span className="font-normal text-muted-foreground">"{ivr.detected_phrase}"</span>
                          </p>
                          <div className="flex items-center gap-2 mt-1">
                            {getActionBadge(ivr.preferred_action)}
                            {ivr.action_value && (
                              <span className="text-sm font-mono bg-muted px-2 py-0.5 rounded">
                                {ivr.preferred_action === 'dtmf' ? `Press ${ivr.action_value}` : ivr.action_value}
                              </span>
                            )}
                            {ivr.success_rate !== undefined && ivr.success_rate > 0 && (
                              <span className="text-xs text-muted-foreground">
                                ({Math.round(ivr.success_rate * 100)}% success)
                              </span>
                            )}
                          </div>
                        </div>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => onDeleteIVR(ivr.id!)}
                        >
                          <Trash2 className="h-3 w-3 text-destructive" />
                        </Button>
                      </div>
                    ))}
                </div>
              ) : (
                <div className="text-sm text-muted-foreground py-4 text-center border rounded-lg bg-background">
                  <Hash className="h-5 w-5 mx-auto mb-1 opacity-50" />
                  No IVR steps configured. Add steps to enable automatic menu navigation.
                </div>
              )}

              {provider.notes && (
                <div className="text-xs text-muted-foreground mt-2">
                  <strong>Notes:</strong> {provider.notes}
                </div>
              )}
            </div>
          </TableCell>
        </TableRow>
      )}
    </>
  )
}
