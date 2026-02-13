'use client'

import { useParams, useRouter } from 'next/navigation'
import { useQuery } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import {
  ArrowLeft,
  Phone,
  Building2,
  FileText,
  Clock,
  Hash,
  MapPin,
  MessageSquare,
  AlertTriangle,
  Loader2,
  CheckCircle2,
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { api } from '@/lib/api'
import { getStatusColor, formatStatus, formatDate, formatPhoneNumber, formatRelativeTime } from '@/lib/utils'
import Link from 'next/link'

export default function CallDetailPage() {
  const params = useParams()
  const router = useRouter()
  const callId = params.id as string

  const { data, isLoading, error } = useQuery({
    queryKey: ['call-detail', callId],
    queryFn: () => api.getCallDetail(callId),
    refetchInterval: 10000,
  })

  const call = data?.data

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    )
  }

  if (error || !call) {
    return (
      <div>
        <Button asChild variant="ghost" size="sm" className="mb-6 gap-1.5">
          <Link href="/calls">
            <ArrowLeft className="h-4 w-4" />
            Back to Calls
          </Link>
        </Button>
        <div className="flex flex-col items-center justify-center py-20 text-center">
          <AlertTriangle className="h-10 w-10 text-muted-foreground mb-4" />
          <p className="text-lg font-medium">Call not found</p>
          <p className="text-sm text-muted-foreground mt-1">
            The call you are looking for does not exist or has been removed.
          </p>
        </div>
      </div>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <Button asChild variant="ghost" size="sm" className="mb-6 gap-1.5">
        <Link href="/calls">
          <ArrowLeft className="h-4 w-4" />
          Back to Calls
        </Link>
      </Button>

      <div className="flex items-start justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold">{call.provider_name}</h1>
          <p className="text-muted-foreground">{call.insurance_name}</p>
        </div>
        <Badge className={getStatusColor(call.status || 'initiated')}>
          {formatStatus(call.status || 'initiated')}
        </Badge>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Left column - Call Info + Questions */}
        <div className="lg:col-span-1 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Building2 className="h-4 w-4" />
                Call Information
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex items-start gap-2">
                <Hash className="h-4 w-4 mt-0.5 text-muted-foreground" />
                <div>
                  <p className="text-xs text-muted-foreground">NPI</p>
                  <p className="font-mono text-sm">{call.npi}</p>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <FileText className="h-4 w-4 mt-0.5 text-muted-foreground" />
                <div>
                  <p className="text-xs text-muted-foreground">Tax ID</p>
                  <p className="font-mono text-sm">{call.tax_id}</p>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <MapPin className="h-4 w-4 mt-0.5 text-muted-foreground" />
                <div>
                  <p className="text-xs text-muted-foreground">Address</p>
                  <p className="text-sm">{call.address}</p>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <Phone className="h-4 w-4 mt-0.5 text-muted-foreground" />
                <div>
                  <p className="text-xs text-muted-foreground">Insurance Phone</p>
                  <p className="text-sm">{formatPhoneNumber(call.insurance_phone)}</p>
                </div>
              </div>
              {call.reference_number && (
                <div className="flex items-start gap-2">
                  <Hash className="h-4 w-4 mt-0.5 text-muted-foreground" />
                  <div>
                    <p className="text-xs text-muted-foreground">Reference Number</p>
                    <p className="font-mono text-sm font-medium">{call.reference_number}</p>
                  </div>
                </div>
              )}
              {call.turnaround_days && (
                <div className="flex items-start gap-2">
                  <Clock className="h-4 w-4 mt-0.5 text-muted-foreground" />
                  <div>
                    <p className="text-xs text-muted-foreground">Turnaround Time</p>
                    <p className="text-sm">{call.turnaround_days} days</p>
                  </div>
                </div>
              )}
              <div className="pt-2 border-t space-y-1">
                {call.created_at && (
                  <p className="text-xs text-muted-foreground">
                    Created: {formatDate(call.created_at)}
                  </p>
                )}
                {call.completed_at && (
                  <p className="text-xs text-muted-foreground">
                    Completed: {formatDate(call.completed_at)}
                  </p>
                )}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <FileText className="h-4 w-4" />
                Questions Asked
              </CardTitle>
            </CardHeader>
            <CardContent>
              {call.questions && call.questions.length > 0 ? (
                <ol className="space-y-2">
                  {call.questions.map((q: string, i: number) => (
                    <li key={i} className="flex items-start gap-2 text-sm">
                      <span className="flex-shrink-0 w-5 h-5 rounded-full bg-primary/15 text-primary text-xs flex items-center justify-center font-medium">
                        {i + 1}
                      </span>
                      <span>{q}</span>
                    </li>
                  ))}
                </ol>
              ) : (
                <p className="text-sm text-muted-foreground">No questions recorded</p>
              )}
            </CardContent>
          </Card>

          {call.missing_documents && call.missing_documents.length > 0 && (
            <Card className="border-yellow-500/30">
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2 text-yellow-400">
                  <AlertTriangle className="h-4 w-4" />
                  Missing Documents
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-1">
                  {call.missing_documents.map((doc: string, i: number) => (
                    <li key={i} className="flex items-center gap-2 text-sm">
                      <span className="h-1.5 w-1.5 rounded-full bg-yellow-500" />
                      {doc}
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Right column - Conversation + Notes */}
        <div className="lg:col-span-2 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <MessageSquare className="h-4 w-4" />
                Conversation Transcript
              </CardTitle>
            </CardHeader>
            <CardContent>
              {call.conversation && call.conversation.length > 0 ? (
                <div className="space-y-3">
                  {call.conversation.map(
                    (msg: { speaker: string; message: string; timestamp?: string }, i: number) => (
                      <div
                        key={i}
                        className={`flex gap-3 ${
                          msg.speaker === 'agent' ? 'justify-end' : 'justify-start'
                        }`}
                      >
                        <div
                          className={`max-w-[80%] rounded-lg px-4 py-2 ${
                            msg.speaker === 'agent'
                              ? 'bg-primary text-primary-foreground'
                              : 'bg-muted'
                          }`}
                        >
                          <div className="flex items-center gap-2 mb-1">
                            <span className="text-xs font-medium opacity-80">
                              {msg.speaker === 'agent' ? 'AI Agent' : 'Representative'}
                            </span>
                            {msg.timestamp && (
                              <span className="text-xs opacity-60">
                                {formatRelativeTime(msg.timestamp)}
                              </span>
                            )}
                          </div>
                          <p className="text-sm">{msg.message}</p>
                        </div>
                      </div>
                    )
                  )}
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center py-8 text-center">
                  <MessageSquare className="h-8 w-8 text-muted-foreground mb-2" />
                  <p className="text-sm text-muted-foreground">No conversation recorded</p>
                  <p className="text-xs text-muted-foreground mt-1">
                    Conversation messages will appear here during and after calls
                  </p>
                </div>
              )}
            </CardContent>
          </Card>

          {call.notes && (
            <Card>
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4" />
                  Notes & Extracted Information
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm whitespace-pre-wrap">{call.notes}</p>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </motion.div>
  )
}
