'use client'

import { useEffect, useState } from 'react'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import * as z from 'zod'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Loader2, ChevronDown, Check, Phone, Hash, Play } from 'lucide-react'
import { api } from '@/lib/api'
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
import type { InsuranceProvider, StartCallResponse } from '@/types'

const callFormSchema = z.object({
  insurance_name: z.string().min(1, 'Insurance name is required'),
  provider_name: z.string().min(1, 'Provider name is required'),
  npi: z.string().trim().regex(/^\d{10}$/, 'NPI must be 10 digits'),
  tax_id: z.string().min(1, 'Tax ID is required'),
  address: z.string().min(1, 'Address is required'),
  insurance_phone: z.string()
    .transform((val) => val.replace(/[\s\-\(\)\.]/g, ''))
    .pipe(z.string().regex(/^\+?1?\d{10,11}$/, 'Valid phone number required (10-11 digits)')),
  provider_phone: z.string()
    .transform((val) => val.replace(/[\s\-\(\)\.]/g, ''))
    .pipe(z.string().regex(/^\+?1?\d{10,11}$/, 'Valid phone number required'))
    .optional()
    .or(z.literal('')),
  questions: z.string().min(1, 'At least one question is required'),
})

type CallFormValues = z.infer<typeof callFormSchema>

interface StartCallFormProps {
  isActive?: boolean
  onCancel?: () => void
  onSuccess?: (response: StartCallResponse) => void
}

export function StartCallForm({
  isActive = true,
  onCancel,
  onSuccess,
}: StartCallFormProps) {
  const queryClient = useQueryClient()
  const [showDropdown, setShowDropdown] = useState(false)
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedInsurance, setSelectedInsurance] = useState<InsuranceProvider | null>(null)

  const form = useForm<CallFormValues>({
    resolver: zodResolver(callFormSchema),
    mode: 'onChange',
    defaultValues: {
      insurance_name: '',
      provider_name: '',
      npi: '',
      tax_id: '',
      address: '',
      insurance_phone: '',
      provider_phone: '',
      questions: '',
    },
  })

  const { data: providers } = useQuery({
    queryKey: ['insurance-providers'],
    queryFn: () => api.getInsuranceProviders(),
    enabled: isActive,
  })

  const { data: ivrKnowledge, isLoading: loadingIVR } = useQuery({
    queryKey: ['ivr-knowledge', selectedInsurance?.insurance_name],
    queryFn: () => api.getIVRKnowledge(selectedInsurance!.insurance_name),
    enabled: isActive && !!selectedInsurance?.insurance_name,
  })

  useEffect(() => {
    if (!isActive) {
      form.reset()
      setSelectedInsurance(null)
      setSearchTerm('')
      setShowDropdown(false)
    }
  }, [isActive, form])

  const filteredProviders = providers?.data?.filter((p) =>
    p.insurance_name.toLowerCase().includes(searchTerm.toLowerCase())
  ) || []

  const handleSelectInsurance = (provider: InsuranceProvider) => {
    setSelectedInsurance(provider)
    form.setValue('insurance_name', provider.insurance_name, { shouldValidate: true })
    form.setValue('insurance_phone', provider.phone_number, { shouldValidate: true })
    setSearchTerm(provider.insurance_name)
    setShowDropdown(false)
  }

  const handleInsuranceInputChange = (value: string) => {
    setSearchTerm(value)
    form.setValue('insurance_name', value, { shouldValidate: true })
    setShowDropdown(true)

    const matchedProvider = providers?.data?.find(
      (p) => p.insurance_name.toLowerCase() === value.toLowerCase()
    )
    if (matchedProvider) {
      setSelectedInsurance(matchedProvider)
      form.setValue('insurance_phone', matchedProvider.phone_number, { shouldValidate: true })
    } else {
      setSelectedInsurance(null)
    }
  }

  const mutation = useMutation({
    mutationFn: (values: CallFormValues) => {
      const questionsArray = values.questions
        .split('\n')
        .filter((q) => q.trim().length > 0)

      return api.startCall({
        ...values,
        questions: questionsArray,
      })
    },
    onSuccess: (response) => {
      toast.success('Call started successfully!', {
        description: `Call ID: ${response.call_id || 'N/A'} - The credentialing call has been initiated.`,
      })
      queryClient.invalidateQueries({ queryKey: ['calls'] })
      queryClient.invalidateQueries({ queryKey: ['metrics'] })
      onSuccess?.(response)
      if (!onSuccess) {
        form.reset()
      }
    },
    onError: (error: any) => {
      toast.error('Failed to start call', {
        description: error.response?.data?.error || error.message,
      })
    },
  })

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

  function onSubmit(values: CallFormValues) {
    mutation.mutate(values)
  }

  return (
    <Form {...form}>
      <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
        <div className="grid gap-4 md:grid-cols-2">
          <FormField
            control={form.control}
            name="insurance_name"
            render={() => (
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
                    <ChevronDown className="absolute right-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                  </div>
                </FormControl>
                {showDropdown && filteredProviders.length > 0 && (
                  <div className="absolute z-50 mt-1 max-h-48 w-full overflow-auto rounded-md border bg-background shadow-lg">
                    {filteredProviders.map((provider) => (
                      <div
                        key={provider.id}
                        className="flex cursor-pointer items-center justify-between px-3 py-2 hover:bg-muted"
                        onMouseDown={() => handleSelectInsurance(provider)}
                      >
                        <div>
                          <div className="font-medium">{provider.insurance_name}</div>
                          <div className="flex items-center gap-1 text-xs text-muted-foreground">
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
            name="provider_phone"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Provider Callback Phone (Optional)</FormLabel>
                <FormControl>
                  <Input placeholder="+18005551234" {...field} />
                </FormControl>
                <FormDescription>
                  Provider&apos;s callback number (will be stated in opening greeting)
                </FormDescription>
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

        {selectedInsurance && (
          <div className="rounded-lg border bg-muted/30 p-4">
            <div className="mb-3 flex items-center gap-2">
              <Phone className="h-4 w-4 text-primary" />
              <h4 className="text-sm font-medium">IVR Navigation (Automatic)</h4>
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
                  .map((ivr) => (
                    <div key={ivr.id} className="flex items-start gap-2 text-sm">
                      <div className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-primary/10 text-xs font-bold text-primary">
                        {ivr.menu_level}
                      </div>
                      <div className="flex-1">
                        <span className="text-muted-foreground">Listen for:</span>{' '}
                        <span className="font-medium">&quot;{ivr.detected_phrase}&quot;</span>
                        <div className="mt-0.5 flex items-center gap-1">
                          {getActionBadge(ivr.preferred_action)}
                          {ivr.action_value && (
                            <span className="rounded border bg-background px-1.5 py-0.5 font-mono text-xs">
                              {ivr.preferred_action === 'dtmf' ? ivr.action_value : `"${ivr.action_value}"`}
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                <div className="mt-2 border-t pt-2 text-xs text-muted-foreground">
                  System will automatically navigate through this menu sequence
                </div>
              </div>
            ) : (
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
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
          {onCancel && (
            <Button
              type="button"
              variant="outline"
              onClick={onCancel}
              disabled={mutation.isPending}
            >
              Cancel
            </Button>
          )}
          <Button type="submit" disabled={mutation.isPending}>
            {mutation.isPending ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            ) : (
              <Play className="mr-2 h-4 w-4" />
            )}
            Start Call
          </Button>
        </div>
      </form>
    </Form>
  )
}
