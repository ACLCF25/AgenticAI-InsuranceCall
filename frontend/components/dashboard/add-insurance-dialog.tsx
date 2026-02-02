// components/dashboard/add-insurance-dialog.tsx
'use client'

import { useEffect } from 'react'
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
import { toast } from 'sonner'
import { Loader2 } from 'lucide-react'
import type { InsuranceProvider } from '@/types'

const insuranceFormSchema = z.object({
  insurance_name: z.string().min(1, 'Insurance name is required'),
  phone_number: z.string()
    .transform(val => val.replace(/[\s\-\(\)\.]/g, ''))
    .pipe(z.string().regex(/^\+?1?\d{10,11}$/, 'Valid phone number required (10-11 digits)')),
  department: z.string().optional(),
  average_wait_time_minutes: z.string().optional().transform(val => val ? parseInt(val, 10) : undefined),
  notes: z.string().optional(),
})

type InsuranceFormValues = z.infer<typeof insuranceFormSchema>

interface AddInsuranceDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  editingProvider?: InsuranceProvider | null
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
      notes: '',
    },
  })

  // Reset form when dialog opens/closes or editing provider changes
  useEffect(() => {
    if (open) {
      if (editingProvider) {
        form.reset({
          insurance_name: editingProvider.insurance_name,
          phone_number: editingProvider.phone_number,
          department: editingProvider.department || '',
          average_wait_time_minutes: editingProvider.average_wait_time_minutes?.toString() || '',
          notes: editingProvider.notes || '',
        })
      } else {
        form.reset({
          insurance_name: '',
          phone_number: '',
          department: '',
          average_wait_time_minutes: '',
          notes: '',
        })
      }
    }
  }, [open, editingProvider, form])

  const addMutation = useMutation({
    mutationFn: (values: InsuranceFormValues) => {
      const payload = {
        insurance_name: values.insurance_name,
        phone_number: values.phone_number,
        department: values.department || undefined,
        average_wait_time_minutes: values.average_wait_time_minutes || undefined,
        notes: values.notes || undefined,
      }
      return api.addInsuranceProvider(payload)
    },
    onSuccess: () => {
      toast.success('Insurance provider added successfully')
      queryClient.invalidateQueries({ queryKey: ['insurance-providers'] })
      onOpenChange(false)
      form.reset()
    },
    onError: (error: any) => {
      toast.error('Failed to add insurance provider', {
        description: error.response?.data?.error || error.message,
      })
    },
  })

  const updateMutation = useMutation({
    mutationFn: (values: InsuranceFormValues) => {
      const payload = {
        insurance_name: values.insurance_name,
        phone_number: values.phone_number,
        department: values.department || undefined,
        average_wait_time_minutes: values.average_wait_time_minutes || undefined,
        notes: values.notes || undefined,
      }
      return api.updateInsuranceProvider(editingProvider!.id!, payload)
    },
    onSuccess: () => {
      toast.success('Insurance provider updated successfully')
      queryClient.invalidateQueries({ queryKey: ['insurance-providers'] })
      onOpenChange(false)
      form.reset()
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
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>{isEditing ? 'Edit Insurance Provider' : 'Add Insurance Provider'}</DialogTitle>
          <DialogDescription>
            {isEditing
              ? 'Update the insurance provider details.'
              : 'Add a new insurance company. You can configure IVR menu steps after adding.'}
          </DialogDescription>
        </DialogHeader>

        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
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
