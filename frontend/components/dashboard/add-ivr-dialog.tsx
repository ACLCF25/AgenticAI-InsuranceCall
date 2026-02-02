// components/dashboard/add-ivr-dialog.tsx
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
import { toast } from 'sonner'
import { Loader2 } from 'lucide-react'

const ivrFormSchema = z.object({
  menu_level: z.string().min(1, 'Menu level is required').transform(val => parseInt(val, 10)),
  detected_phrase: z.string().min(1, 'Detected phrase is required'),
  preferred_action: z.enum(['dtmf', 'speech', 'wait'], {
    required_error: 'Please select an action type',
  }),
  action_value: z.string().optional(),
})

type IVRFormValues = z.infer<typeof ivrFormSchema>

interface AddIVRDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  insuranceName: string
}

export function AddIVRDialog({ open, onOpenChange, insuranceName }: AddIVRDialogProps) {
  const queryClient = useQueryClient()

  const form = useForm<IVRFormValues>({
    resolver: zodResolver(ivrFormSchema),
    defaultValues: {
      menu_level: '1',
      detected_phrase: '',
      preferred_action: 'dtmf',
      action_value: '',
    },
  })

  // Reset form when dialog opens
  useEffect(() => {
    if (open) {
      form.reset({
        menu_level: '1',
        detected_phrase: '',
        preferred_action: 'dtmf',
        action_value: '',
      })
    }
  }, [open, form])

  const mutation = useMutation({
    mutationFn: (values: IVRFormValues) => {
      return api.addIVRKnowledge({
        insurance_name: insuranceName,
        menu_level: values.menu_level,
        detected_phrase: values.detected_phrase,
        preferred_action: values.preferred_action,
        action_value: values.action_value || undefined,
      })
    },
    onSuccess: () => {
      toast.success('IVR step added successfully')
      queryClient.invalidateQueries({ queryKey: ['ivr-knowledge', insuranceName] })
      onOpenChange(false)
      form.reset()
    },
    onError: (error: any) => {
      toast.error('Failed to add IVR step', {
        description: error.response?.data?.error || error.message,
      })
    },
  })

  function onSubmit(values: IVRFormValues) {
    mutation.mutate(values)
  }

  const selectedAction = form.watch('preferred_action')

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>Add IVR Step</DialogTitle>
          <DialogDescription>
            Add a new IVR menu navigation step for <strong>{insuranceName}</strong>.
            The system will listen for the phrase and automatically perform the action.
          </DialogDescription>
        </DialogHeader>

        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
            <FormField
              control={form.control}
              name="menu_level"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Menu Level *</FormLabel>
                  <FormControl>
                    <Input type="number" min="1" placeholder="1" {...field} />
                  </FormControl>
                  <FormDescription>
                    Order of execution (1 = first menu, 2 = submenu, etc.)
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="detected_phrase"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Detected Phrase *</FormLabel>
                  <FormControl>
                    <Input
                      placeholder="press 1 for provider services"
                      {...field}
                    />
                  </FormControl>
                  <FormDescription>
                    What the IVR says (partial match works)
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="preferred_action"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Action Type *</FormLabel>
                  <FormControl>
                    <div className="flex gap-2">
                      <Button
                        type="button"
                        variant={field.value === 'dtmf' ? 'default' : 'outline'}
                        className="flex-1"
                        onClick={() => field.onChange('dtmf')}
                      >
                        Press Button (DTMF)
                      </Button>
                      <Button
                        type="button"
                        variant={field.value === 'speech' ? 'default' : 'outline'}
                        className="flex-1"
                        onClick={() => field.onChange('speech')}
                      >
                        Say Phrase
                      </Button>
                      <Button
                        type="button"
                        variant={field.value === 'wait' ? 'default' : 'outline'}
                        className="flex-1"
                        onClick={() => field.onChange('wait')}
                      >
                        Wait
                      </Button>
                    </div>
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

            {selectedAction !== 'wait' && (
              <FormField
                control={form.control}
                name="action_value"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>
                      {selectedAction === 'dtmf' ? 'Button to Press' : 'Phrase to Say'} *
                    </FormLabel>
                    <FormControl>
                      <Input
                        placeholder={selectedAction === 'dtmf' ? '1' : 'credentialing'}
                        {...field}
                      />
                    </FormControl>
                    <FormDescription>
                      {selectedAction === 'dtmf'
                        ? 'The digit(s) to press (e.g., 1, 2, 0)'
                        : 'The word or phrase to say'}
                    </FormDescription>
                    <FormMessage />
                  </FormItem>
                )}
              />
            )}

            <div className="bg-muted/50 p-3 rounded-lg text-sm">
              <strong>Preview:</strong>
              <p className="text-muted-foreground mt-1">
                When system hears "{form.watch('detected_phrase') || '...'}"
                {selectedAction === 'dtmf' && form.watch('action_value') && (
                  <> → Will press: <span className="font-mono">{form.watch('action_value')}</span></>
                )}
                {selectedAction === 'speech' && form.watch('action_value') && (
                  <> → Will say: "{form.watch('action_value')}"</>
                )}
                {selectedAction === 'wait' && (
                  <> → Will wait for next prompt</>
                )}
              </p>
            </div>

            <div className="flex justify-end gap-3 pt-4">
              <Button
                type="button"
                variant="outline"
                onClick={() => onOpenChange(false)}
                disabled={mutation.isPending}
              >
                Cancel
              </Button>
              <Button type="submit" disabled={mutation.isPending}>
                {mutation.isPending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                Add IVR Step
              </Button>
            </div>
          </form>
        </Form>
      </DialogContent>
    </Dialog>
  )
}
