// components/dashboard/add-insurance-dialog.tsx
'use client'

import { useEffect, useState } from 'react'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import * as z from 'zod'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '@/lib/api'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from '@/components/ui/form'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { Badge } from '@/components/ui/badge'
import { toast } from 'sonner'
import { Loader2, Plus, Trash2, Pencil, Check, X, Hash } from 'lucide-react'
import type { InsuranceProvider } from '@/types'

// Local IVR step type for managing steps in dialog state
type LocalIVRStep = {
  _localId: string
  serverId?: string
  menu_level: number
  detected_phrase: string
  preferred_action: 'dtmf' | 'speech' | 'wait'
  action_value?: string
}

type StepFormValues = {
  menu_level: string
  detected_phrase: string
  preferred_action: LocalIVRStep['preferred_action']
  action_value: string
}

const insuranceFormSchema = z.object({
  insurance_name: z.string().min(1, 'Insurance name is required'),
  phone_number: z.string()
    .transform(val => val.replace(/[\s\-\(\)\.]/g, ''))
    .pipe(z.string().regex(/^\+?1?\d{10,11}$/, 'Valid phone number required (10-11 digits)')),
  department: z.string().optional(),
  average_wait_time_minutes: z.string().optional().transform(val => val ? parseInt(val, 10) : undefined),
  ivr_asks_npi: z.boolean().default(false),
  ivr_npi_method: z.enum(['speech', 'dtmf']).default('speech'),
  ivr_asks_tax_id: z.boolean().default(false),
  ivr_tax_id_method: z.enum(['speech', 'dtmf']).default('speech'),
  ivr_tax_id_digits_mode: z.enum(['full', 'last_n']).default('full'),
  ivr_tax_id_digits_to_send: z.string().optional(),
  notes: z.string().optional(),
}).superRefine((values, ctx) => {
  if (values.ivr_asks_tax_id && values.ivr_tax_id_digits_mode === 'last_n') {
    const parsed = Number.parseInt(values.ivr_tax_id_digits_to_send ?? '', 10)
    if (!Number.isInteger(parsed) || parsed < 1 || parsed > 9) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ['ivr_tax_id_digits_to_send'],
        message: 'Enter a digit count from 1 to 9',
      })
    }
  }
})

type InsuranceFormValues = z.input<typeof insuranceFormSchema>

interface AddInsuranceDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  editingProvider?: InsuranceProvider | null
}

// Default values for inline add-step form
const emptyStepForm: StepFormValues = {
  menu_level: '1',
  detected_phrase: '',
  preferred_action: 'dtmf',
  action_value: '',
}

export function AddInsuranceDialog({ open, onOpenChange, editingProvider }: AddInsuranceDialogProps) {
  const queryClient = useQueryClient()
  const isEditing = !!editingProvider

  const form = useForm<InsuranceFormValues>({
    resolver: zodResolver(insuranceFormSchema),
    defaultValues: {
      insurance_name: '',
      phone_number: '',
      department: '',
      average_wait_time_minutes: '',
      ivr_asks_npi: false,
      ivr_npi_method: 'speech',
      ivr_asks_tax_id: false,
      ivr_tax_id_method: 'speech',
      ivr_tax_id_digits_mode: 'full',
      ivr_tax_id_digits_to_send: '',
      notes: '',
    },
  })

  // IVR step local state
  const [ivrSteps, setIvrSteps] = useState<LocalIVRStep[]>([])
  const [originalIvrSteps, setOriginalIvrSteps] = useState<LocalIVRStep[]>([])
  const [isAddingStep, setIsAddingStep] = useState(false)
  const [editingStepId, setEditingStepId] = useState<string | null>(null)
  const [loadingIvrSteps, setLoadingIvrSteps] = useState(false)

  // Inline add-step form state
  const [stepForm, setStepForm] = useState<StepFormValues>(emptyStepForm)
  // Inline edit-step form state
  const [editStepForm, setEditStepForm] = useState<StepFormValues>(emptyStepForm)

  // Reset form when dialog opens/closes or editing provider changes
  useEffect(() => {
    if (open) {
      if (editingProvider) {
        form.reset({
          insurance_name: editingProvider.insurance_name,
          phone_number: editingProvider.phone_number,
          department: editingProvider.department || '',
          average_wait_time_minutes: editingProvider.average_wait_time_minutes?.toString() || '',
          ivr_asks_npi: editingProvider.ivr_asks_npi ?? false,
          ivr_npi_method: editingProvider.ivr_npi_method ?? 'speech',
          ivr_asks_tax_id: editingProvider.ivr_asks_tax_id ?? false,
          ivr_tax_id_method: editingProvider.ivr_tax_id_method ?? 'speech',
          ivr_tax_id_digits_mode: editingProvider.ivr_tax_id_digits_to_send ? 'last_n' : 'full',
          ivr_tax_id_digits_to_send: editingProvider.ivr_tax_id_digits_to_send?.toString() || '',
          notes: editingProvider.notes || '',
        })

        // Fetch existing IVR steps
        setLoadingIvrSteps(true)
        api.getIVRKnowledge(editingProvider.insurance_name)
          .then(response => {
            const steps: LocalIVRStep[] = (response.data || []).map(ivr => ({
              _localId: crypto.randomUUID(),
              serverId: ivr.id,
              menu_level: ivr.menu_level,
              detected_phrase: ivr.detected_phrase,
              preferred_action: ivr.preferred_action,
              action_value: ivr.action_value || undefined,
            }))
            setIvrSteps(steps)
            setOriginalIvrSteps(steps.map(s => ({ ...s })))
          })
          .catch(() => {
            toast.error('Failed to load IVR steps')
            setIvrSteps([])
            setOriginalIvrSteps([])
          })
          .finally(() => setLoadingIvrSteps(false))
      } else {
        form.reset({
          insurance_name: '',
          phone_number: '',
          department: '',
          average_wait_time_minutes: '',
          ivr_asks_npi: false,
          ivr_npi_method: 'speech',
          ivr_asks_tax_id: false,
          ivr_tax_id_method: 'speech',
          ivr_tax_id_digits_mode: 'full',
          ivr_tax_id_digits_to_send: '',
          notes: '',
        })
        setIvrSteps([])
        setOriginalIvrSteps([])
      }
      setIsAddingStep(false)
      setEditingStepId(null)
      setStepForm(emptyStepForm)
      setEditStepForm(emptyStepForm)
    } else {
      setIvrSteps([])
      setOriginalIvrSteps([])
      setLoadingIvrSteps(false)
      setIsAddingStep(false)
      setEditingStepId(null)
      setStepForm(emptyStepForm)
      setEditStepForm(emptyStepForm)
    }
  }, [open, editingProvider, form])

  // Helper: add step to local state
  const handleAddStep = () => {
    const level = parseInt(stepForm.menu_level, 10)
    if (!stepForm.detected_phrase.trim() || isNaN(level) || level < 1) {
      toast.error('Menu level and detected phrase are required')
      return
    }
    if (stepForm.preferred_action !== 'wait' && !stepForm.action_value?.trim()) {
      toast.error('Action value is required for Press Button and Say Phrase actions')
      return
    }
    setIvrSteps(prev => [...prev, {
      _localId: crypto.randomUUID(),
      menu_level: level,
      detected_phrase: stepForm.detected_phrase.trim(),
      preferred_action: stepForm.preferred_action,
      action_value: stepForm.preferred_action === 'wait' ? undefined : stepForm.action_value?.trim() || undefined,
    }])
    setStepForm(emptyStepForm)
    setIsAddingStep(false)
  }

  // Helper: remove step from local state
  const handleRemoveStep = (localId: string) => {
    setIvrSteps(prev => prev.filter(s => s._localId !== localId))
  }

  // Helper: start editing a step
  const handleStartEditStep = (step: LocalIVRStep) => {
    setEditingStepId(step._localId)
    setEditStepForm({
      menu_level: step.menu_level.toString(),
      detected_phrase: step.detected_phrase,
      preferred_action: step.preferred_action,
      action_value: step.action_value || '',
    })
  }

  // Helper: confirm edit of a step
  const handleConfirmEditStep = () => {
    if (!editingStepId) return
    const level = parseInt(editStepForm.menu_level, 10)
    if (!editStepForm.detected_phrase.trim() || isNaN(level) || level < 1) {
      toast.error('Menu level and detected phrase are required')
      return
    }
    if (editStepForm.preferred_action !== 'wait' && !editStepForm.action_value?.trim()) {
      toast.error('Action value is required for Press Button and Say Phrase actions')
      return
    }
    setIvrSteps(prev => prev.map(s =>
      s._localId === editingStepId
        ? {
            ...s,
            menu_level: level,
            detected_phrase: editStepForm.detected_phrase.trim(),
            preferred_action: editStepForm.preferred_action,
            action_value: editStepForm.preferred_action === 'wait' ? undefined : editStepForm.action_value?.trim() || undefined,
          }
        : s
    ))
    setEditingStepId(null)
    setEditStepForm(emptyStepForm)
  }

  // Helper: action badge
  const getActionBadge = (action: string) => {
    switch (action) {
      case 'dtmf':
        return <Badge variant="secondary" className="text-xs">Press Button</Badge>
      case 'speech':
        return <Badge variant="outline" className="text-xs">Say Phrase</Badge>
      case 'wait':
        return <Badge className="text-xs">Wait</Badge>
      default:
        return <Badge variant="secondary" className="text-xs">{action}</Badge>
    }
  }

  const addMutation = useMutation({
    mutationFn: async (values: InsuranceFormValues) => {
      const parsedTaxDigits = Number.parseInt(values.ivr_tax_id_digits_to_send || '', 10)
      const taxDigitsToSend =
        values.ivr_asks_tax_id && values.ivr_tax_id_digits_mode === 'last_n' && Number.isInteger(parsedTaxDigits)
          ? parsedTaxDigits
          : undefined

      const payload = {
        insurance_name: values.insurance_name,
        phone_number: values.phone_number,
        department: values.department || undefined,
        average_wait_time_minutes: values.average_wait_time_minutes ? parseInt(values.average_wait_time_minutes, 10) : undefined,
        ivr_asks_npi: values.ivr_asks_npi,
        ivr_npi_method: values.ivr_asks_npi ? values.ivr_npi_method : 'speech',
        ivr_asks_tax_id: values.ivr_asks_tax_id,
        ivr_tax_id_method: values.ivr_asks_tax_id ? values.ivr_tax_id_method : 'speech',
        ivr_tax_id_digits_to_send: taxDigitsToSend,
        notes: values.notes || undefined,
      }
      const result = await api.addInsuranceProvider(payload)

      // Create IVR steps
      if (ivrSteps.length > 0) {
        let failedSteps = 0
        for (const step of ivrSteps) {
          try {
            await api.addIVRKnowledge({
              insurance_name: values.insurance_name,
              menu_level: step.menu_level,
              detected_phrase: step.detected_phrase,
              preferred_action: step.preferred_action,
              action_value: step.action_value || undefined,
            })
          } catch {
            failedSteps++
          }
        }
        if (failedSteps > 0) {
          toast.warning(`${failedSteps} IVR step(s) failed to save. You can add them by editing the provider.`)
        }
      }

      return result
    },
    onSuccess: () => {
      toast.success('Insurance provider added successfully')
      queryClient.invalidateQueries({ queryKey: ['insurance-providers'] })
      queryClient.invalidateQueries({ queryKey: ['ivr-knowledge'] })
      onOpenChange(false)
      form.reset()
      setIvrSteps([])
    },
    onError: (error: any) => {
      toast.error('Failed to add insurance provider', {
        description: error.response?.data?.error || error.message,
      })
    },
  })

  const updateMutation = useMutation({
    mutationFn: async (values: InsuranceFormValues) => {
      const parsedTaxDigits = Number.parseInt(values.ivr_tax_id_digits_to_send || '', 10)
      const taxDigitsToSend =
        values.ivr_asks_tax_id && values.ivr_tax_id_digits_mode === 'last_n' && Number.isInteger(parsedTaxDigits)
          ? parsedTaxDigits
          : undefined

      const payload = {
        insurance_name: values.insurance_name,
        phone_number: values.phone_number,
        department: values.department || undefined,
        average_wait_time_minutes: values.average_wait_time_minutes ? parseInt(values.average_wait_time_minutes, 10) : undefined,
        ivr_asks_npi: values.ivr_asks_npi,
        ivr_npi_method: values.ivr_asks_npi ? values.ivr_npi_method : 'speech',
        ivr_asks_tax_id: values.ivr_asks_tax_id,
        ivr_tax_id_method: values.ivr_asks_tax_id ? values.ivr_tax_id_method : 'speech',
        ivr_tax_id_digits_to_send: taxDigitsToSend,
        notes: values.notes || undefined,
      }
      await api.updateInsuranceProvider(editingProvider!.id!, payload)

      // Diff IVR steps
      const insuranceName = values.insurance_name
      const currentServerIds = new Set(ivrSteps.filter(s => s.serverId).map(s => s.serverId!))

      // Steps to delete: in original but not in current
      const stepsToDelete = originalIvrSteps.filter(s => s.serverId && !currentServerIds.has(s.serverId))

      // Steps to add: no serverId (new)
      const stepsToAdd = ivrSteps.filter(s => !s.serverId)

      // Steps to update: have serverId and changed
      const stepsToUpdate = ivrSteps.filter(s => {
        if (!s.serverId) return false
        const original = originalIvrSteps.find(o => o.serverId === s.serverId)
        if (!original) return false
        return (
          original.menu_level !== s.menu_level ||
          original.detected_phrase !== s.detected_phrase ||
          original.preferred_action !== s.preferred_action ||
          (original.action_value || '') !== (s.action_value || '')
        )
      })

      // Execute deletes and updates in parallel
      await Promise.all([
        ...stepsToDelete.map(s => api.deleteIVRKnowledge(s.serverId!)),
        ...stepsToUpdate.map(s => api.updateIVRKnowledge(s.serverId!, {
          menu_level: s.menu_level,
          detected_phrase: s.detected_phrase,
          preferred_action: s.preferred_action,
          action_value: s.action_value || undefined,
        })),
      ])

      // Execute adds
      for (const step of stepsToAdd) {
        await api.addIVRKnowledge({
          insurance_name: insuranceName,
          menu_level: step.menu_level,
          detected_phrase: step.detected_phrase,
          preferred_action: step.preferred_action,
          action_value: step.action_value || undefined,
        })
      }
    },
    onSuccess: () => {
      toast.success('Insurance provider updated successfully')
      queryClient.invalidateQueries({ queryKey: ['insurance-providers'] })
      queryClient.invalidateQueries({ queryKey: ['ivr-knowledge'] })
      onOpenChange(false)
      form.reset()
      setIvrSteps([])
      setOriginalIvrSteps([])
    },
    onError: (error: any) => {
      toast.error('Failed to update insurance provider', {
        description: error.response?.data?.error || error.message,
      })
    },
  })

  function onSubmit(values: InsuranceFormValues) {
    if (isEditing) {
      updateMutation.mutate(values)
    } else {
      addMutation.mutate(values)
    }
  }

  const isPending = addMutation.isPending || updateMutation.isPending

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl max-h-[85vh] flex flex-col">
        <DialogHeader>
          <DialogTitle>{isEditing ? 'Edit Insurance Provider' : 'Add Insurance Provider'}</DialogTitle>
          <DialogDescription>
            {isEditing
              ? 'Update the insurance provider details and IVR navigation steps.'
              : 'Add a new insurance company with optional IVR menu navigation steps.'}
          </DialogDescription>
        </DialogHeader>

        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4 overflow-y-auto flex-1 pr-1">
            <FormField
              control={form.control}
              name="insurance_name"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Insurance Name *</FormLabel>
                  <FormControl>
                    <Input placeholder="Blue Cross Blue Shield" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="phone_number"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Phone Number *</FormLabel>
                  <FormControl>
                    <Input placeholder="+18001234567" {...field} />
                  </FormControl>
                  <FormDescription>Include country code</FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="department"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Department</FormLabel>
                  <FormControl>
                    <Input placeholder="Provider Services / Credentialing" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

            {/* IVR Auto-Response Configuration */}
            <div className="space-y-3 rounded-md border border-border p-3">
              <p className="text-sm font-medium">IVR Auto-Response</p>
              <p className="text-xs text-muted-foreground">
                If this insurance&apos;s IVR asks for these during menu navigation, the system will automatically respond.
              </p>

              {/* NPI Checkbox + Method */}
              <FormField
                control={form.control}
                name="ivr_asks_npi"
                render={({ field }) => (
                  <FormItem className="flex items-center gap-2 space-y-0">
                    <FormControl>
                      <input
                        type="checkbox"
                        className="h-4 w-4 rounded border-border accent-primary"
                        checked={field.value}
                        onChange={field.onChange}
                      />
                    </FormControl>
                    <FormLabel className="text-sm font-normal">IVR asks for NPI number</FormLabel>
                  </FormItem>
                )}
              />
              {form.watch('ivr_asks_npi') && (
                <FormField
                  control={form.control}
                  name="ivr_npi_method"
                  render={({ field }) => (
                    <FormItem className="ml-6">
                      <div className="flex items-center gap-4">
                        <label className="flex items-center gap-1.5 text-sm">
                          <input
                            type="radio"
                            name="ivr_npi_method"
                            value="speech"
                            checked={field.value === 'speech'}
                            onChange={() => field.onChange('speech')}
                            className="accent-primary"
                          />
                          Speak it
                        </label>
                        <label className="flex items-center gap-1.5 text-sm">
                          <input
                            type="radio"
                            name="ivr_npi_method"
                            value="dtmf"
                            checked={field.value === 'dtmf'}
                            onChange={() => field.onChange('dtmf')}
                            className="accent-primary"
                          />
                          Enter via keypad
                        </label>
                      </div>
                    </FormItem>
                  )}
                />
              )}

              {/* Tax ID Checkbox + Method */}
              <FormField
                control={form.control}
                name="ivr_asks_tax_id"
                render={({ field }) => (
                  <FormItem className="flex items-center gap-2 space-y-0">
                    <FormControl>
                      <input
                        type="checkbox"
                        className="h-4 w-4 rounded border-border accent-primary"
                        checked={field.value}
                        onChange={field.onChange}
                      />
                    </FormControl>
                    <FormLabel className="text-sm font-normal">IVR asks for Tax ID number</FormLabel>
                  </FormItem>
                )}
              />
              {form.watch('ivr_asks_tax_id') && (
                <div className="ml-6 space-y-3">
                  <FormField
                    control={form.control}
                    name="ivr_tax_id_method"
                    render={({ field }) => (
                      <FormItem>
                        <div className="flex items-center gap-4">
                          <label className="flex items-center gap-1.5 text-sm">
                            <input
                              type="radio"
                              name="ivr_tax_id_method"
                              value="speech"
                              checked={field.value === 'speech'}
                              onChange={() => field.onChange('speech')}
                              className="accent-primary"
                            />
                            Speak it
                          </label>
                          <label className="flex items-center gap-1.5 text-sm">
                            <input
                              type="radio"
                              name="ivr_tax_id_method"
                              value="dtmf"
                              checked={field.value === 'dtmf'}
                              onChange={() => field.onChange('dtmf')}
                              className="accent-primary"
                            />
                            Enter via keypad
                          </label>
                        </div>
                      </FormItem>
                    )}
                  />

                  <FormField
                    control={form.control}
                    name="ivr_tax_id_digits_mode"
                    render={({ field }) => (
                      <FormItem>
                        <p className="text-xs text-muted-foreground">
                          NPI is always sent as full 10 digits. Tax ID can be sent as full or last N digits.
                        </p>
                        <div className="flex items-center gap-4 pt-1">
                          <label className="flex items-center gap-1.5 text-sm">
                            <input
                              type="radio"
                              name="ivr_tax_id_digits_mode"
                              value="full"
                              checked={field.value === 'full'}
                              onChange={() => field.onChange('full')}
                              className="accent-primary"
                            />
                            Send full Tax ID
                          </label>
                          <label className="flex items-center gap-1.5 text-sm">
                            <input
                              type="radio"
                              name="ivr_tax_id_digits_mode"
                              value="last_n"
                              checked={field.value === 'last_n'}
                              onChange={() => field.onChange('last_n')}
                              className="accent-primary"
                            />
                            Send last N digits
                          </label>
                        </div>
                      </FormItem>
                    )}
                  />

                  {form.watch('ivr_tax_id_digits_mode') === 'last_n' && (
                    <FormField
                      control={form.control}
                      name="ivr_tax_id_digits_to_send"
                      render={({ field }) => (
                        <FormItem className="max-w-[140px]">
                          <FormLabel className="text-xs">Digits to send (1-9)</FormLabel>
                          <FormControl>
                            <Input
                              type="number"
                              min={1}
                              max={9}
                              placeholder="4"
                              {...field}
                            />
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                  )}
                </div>
              )}
            </div>

            {/* IVR Menu Navigation Steps */}
            <div className="space-y-3 rounded-md border border-border p-3">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium">IVR Menu Navigation Steps</p>
                  <p className="text-xs text-muted-foreground">
                    Define actions for navigating this insurance&apos;s phone menu.
                  </p>
                </div>
                {!isAddingStep && (
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      setStepForm(emptyStepForm)
                      setIsAddingStep(true)
                    }}
                  >
                    <Plus className="mr-1 h-3 w-3" />
                    Add Step
                  </Button>
                )}
              </div>

              {/* Loading state */}
              {loadingIvrSteps && (
                <div className="flex items-center gap-2 text-sm text-muted-foreground py-2">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Loading IVR steps...
                </div>
              )}

              {/* Steps list */}
              {!loadingIvrSteps && ivrSteps.length > 0 && (
                <div className="space-y-2 max-h-[240px] overflow-y-auto">
                  {[...ivrSteps]
                    .sort((a, b) => a.menu_level - b.menu_level)
                    .map((step) => (
                      editingStepId === step._localId ? (
                        // Inline edit form
                        <div key={step._localId} className="flex flex-wrap items-start gap-2 p-3 border border-primary/30 rounded-lg bg-muted/30">
                          <Input
                            type="number"
                            min={1}
                            placeholder="Level"
                            className="w-16 h-8 text-sm"
                            value={editStepForm.menu_level}
                            onChange={e => setEditStepForm(prev => ({ ...prev, menu_level: e.target.value }))}
                          />
                          <Input
                            placeholder="Detected phrase..."
                            className="flex-1 min-w-[140px] h-8 text-sm"
                            value={editStepForm.detected_phrase}
                            onChange={e => setEditStepForm(prev => ({ ...prev, detected_phrase: e.target.value }))}
                          />
                          <div className="flex gap-1">
                            <Button
                              type="button"
                              size="sm"
                              variant={editStepForm.preferred_action === 'dtmf' ? 'default' : 'outline'}
                              className="h-8 text-xs px-2"
                              onClick={() => setEditStepForm(prev => ({ ...prev, preferred_action: 'dtmf' }))}
                            >
                              DTMF
                            </Button>
                            <Button
                              type="button"
                              size="sm"
                              variant={editStepForm.preferred_action === 'speech' ? 'default' : 'outline'}
                              className="h-8 text-xs px-2"
                              onClick={() => setEditStepForm(prev => ({ ...prev, preferred_action: 'speech' }))}
                            >
                              Speech
                            </Button>
                            <Button
                              type="button"
                              size="sm"
                              variant={editStepForm.preferred_action === 'wait' ? 'default' : 'outline'}
                              className="h-8 text-xs px-2"
                              onClick={() => setEditStepForm(prev => ({ ...prev, preferred_action: 'wait' }))}
                            >
                              Wait
                            </Button>
                          </div>
                          {editStepForm.preferred_action !== 'wait' && (
                            <Input
                              placeholder={editStepForm.preferred_action === 'dtmf' ? 'Key' : 'Phrase'}
                              className="w-20 h-8 text-sm"
                              value={editStepForm.action_value}
                              onChange={e => setEditStepForm(prev => ({ ...prev, action_value: e.target.value }))}
                            />
                          )}
                          <Button type="button" size="sm" className="h-8 w-8 p-0" onClick={handleConfirmEditStep}>
                            <Check className="h-3 w-3" />
                          </Button>
                          <Button
                            type="button"
                            size="sm"
                            variant="ghost"
                            className="h-8 w-8 p-0"
                            onClick={() => { setEditingStepId(null); setEditStepForm(emptyStepForm) }}
                          >
                            <X className="h-3 w-3" />
                          </Button>
                        </div>
                      ) : (
                        // Display row
                        <div key={step._localId} className="flex items-center gap-3 p-2.5 bg-background rounded-lg border">
                          <div className="flex items-center justify-center w-6 h-6 rounded-full bg-primary/10 text-primary text-xs font-bold shrink-0">
                            {step.menu_level}
                          </div>
                          <div className="flex-1 min-w-0">
                            <p className="text-sm truncate">
                              &quot;{step.detected_phrase}&quot;
                            </p>
                          </div>
                          {getActionBadge(step.preferred_action)}
                          {step.action_value && (
                            <span className="text-xs font-mono bg-muted px-2 py-0.5 rounded">
                              {step.preferred_action === 'dtmf' ? `Press ${step.action_value}` : step.action_value}
                            </span>
                          )}
                          <Button
                            type="button"
                            variant="ghost"
                            size="sm"
                            className="h-7 w-7 p-0"
                            onClick={() => handleStartEditStep(step)}
                          >
                            <Pencil className="h-3 w-3" />
                          </Button>
                          <Button
                            type="button"
                            variant="ghost"
                            size="sm"
                            className="h-7 w-7 p-0"
                            onClick={() => handleRemoveStep(step._localId)}
                          >
                            <Trash2 className="h-3 w-3 text-destructive" />
                          </Button>
                        </div>
                      )
                    ))}
                </div>
              )}

              {/* Empty state */}
              {!loadingIvrSteps && ivrSteps.length === 0 && !isAddingStep && (
                <div className="text-sm text-muted-foreground py-3 text-center border rounded-lg bg-background">
                  <Hash className="h-4 w-4 mx-auto mb-1 opacity-50" />
                  No IVR steps configured yet.
                </div>
              )}

              {/* Inline add-step form */}
              {isAddingStep && (
                <div className="flex flex-wrap items-start gap-2 p-3 border border-dashed rounded-lg bg-muted/30">
                  <Input
                    type="number"
                    min={1}
                    placeholder="Level"
                    className="w-16 h-8 text-sm"
                    value={stepForm.menu_level}
                    onChange={e => setStepForm(prev => ({ ...prev, menu_level: e.target.value }))}
                  />
                  <Input
                    placeholder="Detected phrase..."
                    className="flex-1 min-w-[140px] h-8 text-sm"
                    value={stepForm.detected_phrase}
                    onChange={e => setStepForm(prev => ({ ...prev, detected_phrase: e.target.value }))}
                  />
                  <div className="flex gap-1">
                    <Button
                      type="button"
                      size="sm"
                      variant={stepForm.preferred_action === 'dtmf' ? 'default' : 'outline'}
                      className="h-8 text-xs px-2"
                      onClick={() => setStepForm(prev => ({ ...prev, preferred_action: 'dtmf' }))}
                    >
                      DTMF
                    </Button>
                    <Button
                      type="button"
                      size="sm"
                      variant={stepForm.preferred_action === 'speech' ? 'default' : 'outline'}
                      className="h-8 text-xs px-2"
                      onClick={() => setStepForm(prev => ({ ...prev, preferred_action: 'speech' }))}
                    >
                      Speech
                    </Button>
                    <Button
                      type="button"
                      size="sm"
                      variant={stepForm.preferred_action === 'wait' ? 'default' : 'outline'}
                      className="h-8 text-xs px-2"
                      onClick={() => setStepForm(prev => ({ ...prev, preferred_action: 'wait' }))}
                    >
                      Wait
                    </Button>
                  </div>
                  {stepForm.preferred_action !== 'wait' && (
                    <Input
                      placeholder={stepForm.preferred_action === 'dtmf' ? 'Key' : 'Phrase'}
                      className="w-20 h-8 text-sm"
                      value={stepForm.action_value}
                      onChange={e => setStepForm(prev => ({ ...prev, action_value: e.target.value }))}
                    />
                  )}
                  <Button type="button" size="sm" className="h-8 w-8 p-0" onClick={handleAddStep}>
                    <Check className="h-3 w-3" />
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    variant="ghost"
                    className="h-8 w-8 p-0"
                    onClick={() => { setIsAddingStep(false); setStepForm(emptyStepForm) }}
                  >
                    <X className="h-3 w-3" />
                  </Button>
                </div>
              )}
            </div>

            <FormField
              control={form.control}
              name="average_wait_time_minutes"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Average Wait Time (minutes)</FormLabel>
                  <FormControl>
                    <Input type="number" placeholder="15" {...field} />
                  </FormControl>
                  <FormDescription>Typical hold time</FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="notes"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Notes</FormLabel>
                  <FormControl>
                    <Textarea
                      placeholder="Best to call early morning. Ask for credentialing department specifically."
                      className="min-h-[80px]"
                      {...field}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

            <div className="flex justify-end gap-3 pt-4">
              <Button
                type="button"
                variant="outline"
                onClick={() => onOpenChange(false)}
                disabled={isPending}
              >
                Cancel
              </Button>
              <Button type="submit" disabled={isPending}>
                {isPending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                {isEditing ? 'Update' : 'Add Insurance'}
              </Button>
            </div>
          </form>
        </Form>
      </DialogContent>
    </Dialog>
  )
}
