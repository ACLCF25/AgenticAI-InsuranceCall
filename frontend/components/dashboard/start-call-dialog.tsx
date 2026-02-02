// components/dashboard/start-call-dialog.tsx
'use client'

import { useState, useEffect } from 'react'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import * as z from 'zod'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
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
import { Loader2, ChevronDown, Check, Phone, Hash } from 'lucide-react'
import type { InsuranceProvider, IVRKnowledge } from '@/types'

const callFormSchema = z.object({
  insurance_name: z.string().min(1, 'Insurance name is required'),
  provider_name: z.string().min(1, 'Provider name is required'),
  npi: z.string().trim().regex(/^\d{10}$/, 'NPI must be 10 digits'),
  tax_id: z.string().min(1, 'Tax ID is required'),
  address: z.string().min(1, 'Address is required'),
  // Strip spaces, dashes, parentheses, dots before validating phone
  insurance_phone: z.string()
    .transform(val => val.replace(/[\s\-\(\)\.]/g, ''))
    .pipe(z.string().regex(/^\+?1?\d{10,11}$/, 'Valid phone number required (10-11 digits)')),
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
  const [showDropdown, setShowDropdown] = useState(false)
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedInsurance, setSelectedInsurance] = useState<InsuranceProvider | null>(null)

  const form = useForm<CallFormValues>({
    resolver: zodResolver(callFormSchema),
    mode: 'onChange', // Show validation errors as user types
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

  // Fetch insurance providers
  const { data: providers } = useQuery({
    queryKey: ['insurance-providers'],
    queryFn: () => api.getInsuranceProviders(),
    enabled: open,
  })

  // Fetch IVR knowledge when insurance is selected
  const { data: ivrKnowledge, isLoading: loadingIVR } = useQuery({
    queryKey: ['ivr-knowledge', selectedInsurance?.insurance_name],
    queryFn: () => api.getIVRKnowledge(selectedInsurance!.insurance_name),
    enabled: !!selectedInsurance?.insurance_name,
  })

  // Reset form when dialog closes
  useEffect(() => {
    if (!open) {
      form.reset()
      setSelectedInsurance(null)
      setSearchTerm('')
    }
  }, [open, form])

  // Filter providers based on search term
  const filteredProviders = providers?.data?.filter(p =>
    p.insurance_name.toLowerCase().includes(searchTerm.toLowerCase())
  ) || []

  const handleSelectInsurance = (provider: InsuranceProvider) => {
    setSelectedInsurance(provider)
    form.setValue('insurance_name', provider.insurance_name)
    form.setValue('insurance_phone', provider.phone_number)
    setSearchTerm(provider.insurance_name)
    setShowDropdown(false)
  }

  const handleInsuranceInputChange = (value: string) => {
    setSearchTerm(value)
    form.setValue('insurance_name', value)
    setShowDropdown(true)

    // Check if typed value matches an existing provider
    const matchedProvider = providers?.data?.find(
      p => p.insurance_name.toLowerCase() === value.toLowerCase()
    )
    if (matchedProvider) {
      setSelectedInsurance(matchedProvider)
      form.setValue('insurance_phone', matchedProvider.phone_number)
    } else {
      setSelectedInsurance(null)
    }
  }

  const mutation = useMutation({
    mutationFn: (values: CallFormValues) => {
      console.log('Starting call with values:', values)
      // Convert questions string to array
      const questionsArray = values.questions
        .split('\n')
        .filter(q => q.trim().length > 0)

      const payload = {
        ...values,
        questions: questionsArray,
      }
      console.log('Sending API request:', payload)
      return api.startCall(payload)
    },
    onSuccess: (response) => {
      console.log('Call started successfully:', response)
      toast.success('Call started successfully!', {
        description: `Call ID: ${response.call_id || 'N/A'} - The credentialing call has been initiated.`,
      })
      queryClient.invalidateQueries({ queryKey: ['calls'] })
      queryClient.invalidateQueries({ queryKey: ['metrics'] })
      onOpenChange(false)
      form.reset()
    },
    onError: (error: any) => {
      console.error('Call failed:', error)
      console.error('Error details:', error.response?.data)
      toast.error('Failed to start call', {
        description: error.response?.data?.error || error.message,
      })
    },
  })

  function onSubmit(values: CallFormValues) {
    console.log('Form submitted, values:', values)
    mutation.mutate(values)
  }

  // Log validation errors whenever they change
  const errors = form.formState.errors
  if (Object.keys(errors).length > 0) {
    console.log('Form validation errors:', errors)
  }

  const getActionBadge = (action: string) => {
    switch (action) {
      case 'dtmf':
        return <Badge variant="secondary" className="text-xs">Press</Badge>
      case 'speech':
        return <Badge variant="outline" className="text-xs">Say</Badge>
      case 'wait':
        return <Badge className="text-xs">Wait</Badge>
      default:
        return <Badge variant="secondary" className="text-xs">{action}</Badge>
    }
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
              {/* Insurance Name with Dropdown */}
              <FormField
                control={form.control}
                name="insurance_name"
                render={({ field }) => (
                  <FormItem className="relative">
                    <FormLabel>Insurance Name</FormLabel>
                    <FormControl>
                      <div className="relative">
                        <Input
                          placeholder="Search or type insurance name..."
                          value={searchTerm}
                          onChange={(e) => handleInsuranceInputChange(e.target.value)}
                          onFocus={() => setShowDropdown(true)}
                          onBlur={() => setTimeout(() => setShowDropdown(false), 200)}
                        />
                        <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                      </div>
                    </FormControl>
                    {showDropdown && filteredProviders.length > 0 && (
                      <div className="absolute z-50 w-full mt-1 bg-background border rounded-md shadow-lg max-h-48 overflow-auto">
                        {filteredProviders.map((provider) => (
                          <div
                            key={provider.id}
                            className="px-3 py-2 cursor-pointer hover:bg-muted flex items-center justify-between"
                            onMouseDown={() => handleSelectInsurance(provider)}
                          >
                            <div>
                              <div className="font-medium">{provider.insurance_name}</div>
                              <div className="text-xs text-muted-foreground flex items-center gap-1">
                                <Phone className="h-3 w-3" />
                                {provider.phone_number}
                              </div>
                            </div>
                            {selectedInsurance?.id === provider.id && (
                              <Check className="h-4 w-4 text-primary" />
                            )}
                          </div>
                        ))}
                      </div>
                    )}
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
                    <FormDescription>
                      {selectedInsurance ? 'Auto-filled from selected insurance' : 'Include country code'}
                    </FormDescription>
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

            {/* IVR Navigation Preview */}
            {selectedInsurance && (
              <div className="border rounded-lg p-4 bg-muted/30">
                <div className="flex items-center gap-2 mb-3">
                  <Phone className="h-4 w-4 text-primary" />
                  <h4 className="font-medium text-sm">IVR Navigation (Automatic)</h4>
                </div>

                {loadingIVR ? (
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Loading IVR configuration...
                  </div>
                ) : ivrKnowledge?.data && ivrKnowledge.data.length > 0 ? (
                  <div className="space-y-2">
                    {ivrKnowledge.data
                      .sort((a, b) => a.menu_level - b.menu_level)
                      .map((ivr, index) => (
                        <div key={ivr.id} className="flex items-start gap-2 text-sm">
                          <div className="flex items-center justify-center w-5 h-5 rounded-full bg-primary/10 text-primary text-xs font-bold shrink-0 mt-0.5">
                            {ivr.menu_level}
                          </div>
                          <div className="flex-1">
                            <span className="text-muted-foreground">Listen for:</span>{' '}
                            <span className="font-medium">"{ivr.detected_phrase}"</span>
                            <div className="flex items-center gap-1 mt-0.5">
                              {getActionBadge(ivr.preferred_action)}
                              {ivr.action_value && (
                                <span className="font-mono text-xs bg-background px-1.5 py-0.5 rounded border">
                                  {ivr.preferred_action === 'dtmf' ? ivr.action_value : `"${ivr.action_value}"`}
                                </span>
                              )}
                            </div>
                          </div>
                        </div>
                      ))}
                    <div className="text-xs text-muted-foreground mt-2 pt-2 border-t">
                      System will automatically navigate through this menu sequence
                    </div>
                  </div>
                ) : (
                  <div className="text-sm text-muted-foreground flex items-center gap-2">
                    <Hash className="h-4 w-4" />
                    No IVR steps configured for this insurance. The system will attempt to navigate dynamically.
                  </div>
                )}
              </div>
            )}

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
