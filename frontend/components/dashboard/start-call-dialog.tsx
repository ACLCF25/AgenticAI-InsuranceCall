// components/dashboard/start-call-dialog.tsx
'use client'

import { useState } from 'react'
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

const callFormSchema = z.object({
  insurance_name: z.string().min(1, 'Insurance name is required'),
  provider_name: z.string().min(1, 'Provider name is required'),
  npi: z.string().regex(/^\d{10}$/, 'NPI must be 10 digits'),
  tax_id: z.string().min(1, 'Tax ID is required'),
  address: z.string().min(1, 'Address is required'),
  insurance_phone: z.string().regex(/^\+?1?\d{10,11}$/, 'Valid phone number required'),
  questions: z.string().min(1, 'At least one question is required'),
})

type CallFormValues = z.infer<typeof callFormSchema>

interface StartCallDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function StartCallDialog({ open, onOpenChange }: StartCallDialogProps) {
  console.log('StartCallDialog render, open:', open)
  const queryClient = useQueryClient()
  
  const form = useForm<CallFormValues>({
    resolver: zodResolver(callFormSchema),
    defaultValues: {
      insurance_name: '',
      provider_name: '',
      npi: '',
      tax_id: '',
      address: '',
      insurance_phone: '',
      questions: '',
    },
  })

  const mutation = useMutation({
    mutationFn: (values: CallFormValues) => {
      // Convert questions string to array
      const questionsArray = values.questions
        .split('\n')
        .filter(q => q.trim().length > 0)
      
      return api.startCall({
        ...values,
        questions: questionsArray,
      })
    },
    onSuccess: () => {
      toast.success('Call started successfully!', {
        description: 'The credentialing call has been initiated.',
      })
      queryClient.invalidateQueries({ queryKey: ['calls'] })
      queryClient.invalidateQueries({ queryKey: ['metrics'] })
      onOpenChange(false)
      form.reset()
    },
    onError: (error: any) => {
      toast.error('Failed to start call', {
        description: error.response?.data?.error || error.message,
      })
    },
  })

  function onSubmit(values: CallFormValues) {
    mutation.mutate(values)
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Start New Credentialing Call</DialogTitle>
          <DialogDescription>
            Enter the details for the credentialing call. The AI agent will handle the entire process autonomously.
          </DialogDescription>
        </DialogHeader>

        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
            <div className="grid gap-4 md:grid-cols-2">
              <FormField
                control={form.control}
                name="insurance_name"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Insurance Name</FormLabel>
                    <FormControl>
                      <Input placeholder="Blue Cross Blue Shield" {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="provider_name"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Provider Name</FormLabel>
                    <FormControl>
                      <Input placeholder="Dr. Jane Smith" {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="npi"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>NPI Number</FormLabel>
                    <FormControl>
                      <Input placeholder="1234567890" {...field} />
                    </FormControl>
                    <FormDescription>10-digit number</FormDescription>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="tax_id"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Tax ID</FormLabel>
                    <FormControl>
                      <Input placeholder="12-3456789" {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="insurance_phone"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Insurance Phone</FormLabel>
                    <FormControl>
                      <Input placeholder="+18001234567" {...field} />
                    </FormControl>
                    <FormDescription>Include country code</FormDescription>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </div>

            <FormField
              control={form.control}
              name="address"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Provider Address</FormLabel>
                  <FormControl>
                    <Input placeholder="123 Main St, Suite 100, City, ST 12345" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="questions"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Credentialing Questions</FormLabel>
                  <FormControl>
                    <Textarea
                      placeholder="What is the current credentialing status?&#10;Are any additional documents required?&#10;What is the expected completion date?"
                      className="min-h-[120px]"
                      {...field}
                    />
                  </FormControl>
                  <FormDescription>
                    Enter one question per line
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />

            <div className="flex justify-end gap-3">
              <Button
                type="button"
                variant="outline"
                onClick={() => onOpenChange(false)}
                disabled={mutation.isPending}
              >
                Cancel
              </Button>
              <Button type="submit" disabled={mutation.isPending}>
                {mutation.isPending && (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                )}
                Start Call
              </Button>
            </div>
          </form>
        </Form>
      </DialogContent>
    </Dialog>
  )
}
