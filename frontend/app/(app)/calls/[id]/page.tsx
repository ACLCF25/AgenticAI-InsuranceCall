'use client'

import { useParams, useRouter } from 'next/navigation'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import { toast } from 'sonner'
import { useEffect, useState } from 'react'
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
  Bot,
  User,
  ListOrdered,
  Activity,
  ThumbsUp,
  ThumbsDown,
  Brain,
  Sparkles,
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { LocalDateTime } from '@/components/ui/local-date-time'
import { api } from '@/lib/api'
import { getStatusColor, formatStatus, formatPhoneNumber } from '@/lib/utils'
import Link from 'next/link'

function splitTranscriptParagraphs(text: string): string[] {
  const normalized = text.replace(/\s+/g, ' ').trim()
  if (!normalized) return []

  const sentences = normalized.split(/(?<=[.!?])\s+(?=[A-Z0-9])/)
  const paragraphs: string[] = []
  let current = ''
  let sentenceCount = 0

  for (const sentence of sentences) {
    const trimmed = sentence.trim()
    if (!trimmed) continue

    const next = current ? `${current} ${trimmed}` : trimmed
    if (current && (next.length > 420 || sentenceCount >= 3)) {
      paragraphs.push(current)
      current = trimmed
      sentenceCount = 1
    } else {
      current = next
      sentenceCount += 1
    }
  }

  if (current) paragraphs.push(current)
  return paragraphs.length > 0 ? paragraphs : [normalized]
}

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

  // Fetch the recording as a blob so the Authorization header is sent.
  // A plain <audio src="..."> cannot attach JWT tokens, causing a 401 on the
  // protected /api/call-recording/:id/stream endpoint.
  const [recordingBlobUrl, setRecordingBlobUrl] = useState<string | null>(null)
  const [recordingLoadError, setRecordingLoadError] = useState<string | null>(null)
  const queryClient = useQueryClient()

  // Human detection feedback state
  const [feedbackSubmitting, setFeedbackSubmitting] = useState(false)
  const [feedbackResult, setFeedbackResult] = useState<{
    new_phrases: Array<{ phrase: string; phrase_type: string; confidence: number }>;
    analysis?: string;
  } | null>(null)

  useEffect(() => {
    if (call?.recording?.status !== 'completed') {
      setRecordingLoadError(null)
      setRecordingBlobUrl((current) => {
        if (current) URL.revokeObjectURL(current)
        return null
      })
      return
    }

    let objectUrl: string | null = null
    let cancelled = false
    setRecordingLoadError(null)

    api
      .getCallRecordingBlob(callId)
      .then((blob) => {
        if (cancelled) return
        objectUrl = URL.createObjectURL(blob)
        setRecordingBlobUrl((current) => {
          if (current) URL.revokeObjectURL(current)
          return objectUrl
        })
      })
      .catch(() => {
        if (cancelled) return
        setRecordingLoadError('Recording could not be loaded.')
      })

    return () => {
      cancelled = true
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [callId, call?.recording?.status])

  const handleFeedback = async (correct: boolean) => {
    if (!call) return
    setFeedbackSubmitting(true)
    try {
      const res = await api.submitHumanDetectionFeedback(call.id, correct)
      if (!correct && res.new_phrases?.length) {
        setFeedbackResult({ new_phrases: res.new_phrases, analysis: res.analysis })
      } else {
        setFeedbackResult({ new_phrases: [], analysis: correct ? 'Marked as correct' : res.analysis })
      }
      queryClient.invalidateQueries({ queryKey: ['call-detail', callId] })
    } catch (err) {
      console.error('Feedback submission failed:', err)
      toast.error('Failed to submit feedback. Please try again.')
    } finally {
      setFeedbackSubmitting(false)
    }
  }

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
              {call.call_mode && (
              <div className="flex items-start gap-2">
                {call.call_mode === 'agent' ? (
                  <User className="h-4 w-4 mt-0.5 text-muted-foreground" />
                ) : (
                  <Bot className="h-4 w-4 mt-0.5 text-muted-foreground" />
                )}
                <div>
                  <p className="text-xs text-muted-foreground">Call Mode</p>
                  <p className="text-sm capitalize">{call.call_mode === 'agent' ? 'Human Agent' : 'AI Agent'}</p>
                </div>
              </div>
              )}
              {call.call_mode === 'agent' && call.agent_phone && (
                <div className="flex items-start gap-2">
                  <Phone className="h-4 w-4 mt-0.5 text-muted-foreground" />
                  <div>
                    <p className="text-xs text-muted-foreground">Human Agent Phone</p>
                    <p className="text-sm">{formatPhoneNumber(call.agent_phone)}</p>
                  </div>
                </div>
              )}
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
                    Created: <LocalDateTime value={call.created_at} fallback="" />
                  </p>
                )}
                {call.completed_at && (
                  <p className="text-xs text-muted-foreground">
                    Completed: <LocalDateTime value={call.completed_at} fallback="" />
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

          {call.recording && (
            <Card>
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <Phone className="h-4 w-4" />
                  Call Recording
                  {call.recording.recording_type && (
                    <Badge variant={call.recording.recording_type === 'agent' ? 'secondary' : 'outline'} className="text-xs ml-2">
                      {call.recording.recording_type === 'agent' ? 'Agent Transfer' : call.recording.recording_type === 'both' ? 'AI + Agent' : 'AI Call'}
                    </Badge>
                  )}
                </CardTitle>
              </CardHeader>
              <CardContent>
                {call.recording.status === 'failed' ? (
                  <div className="flex items-center gap-2 text-sm text-yellow-600 py-2">
                    <AlertTriangle className="h-4 w-4" />
                    <span>Recording could not be processed.</span>
                  </div>
                ) : call.recording.status !== 'completed' ? (
                  <div className="flex items-center gap-2 text-sm text-muted-foreground py-2">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    <span>
                      {call.recording.status === 'pending'
                        ? 'Recording is pending and will appear after Twilio posts it.'
                        : 'Recording is being processed...'}
                    </span>
                  </div>
                ) : recordingLoadError ? (
                  <div className="flex items-center gap-2 text-sm text-yellow-600 py-2">
                    <AlertTriangle className="h-4 w-4" />
                    <span>{recordingLoadError}</span>
                  </div>
                ) : recordingBlobUrl ? (
                  <>
                    <audio
                      controls
                      className="w-full"
                      src={recordingBlobUrl}
                      preload="metadata"
                    >
                      Your browser does not support audio playback.
                    </audio>
                    <p className="text-xs text-muted-foreground mt-2">
                      Duration: {Math.floor((call.recording.duration || 0) / 60)}:
                      {((call.recording.duration || 0) % 60).toString().padStart(2, '0')}
                    </p>
                  </>
                ) : (
                  <div className="flex items-center gap-2 text-sm text-muted-foreground py-2">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    <span>Loading recording&hellip;</span>
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          {call.qa_pairs && call.qa_pairs.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <MessageSquare className="h-4 w-4" />
                  Questions & Answers
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {call.qa_pairs.map((qa) => (
                  <div key={qa.id} className="border-l-2 border-primary/30 pl-4">
                    <div className="flex items-start gap-2 mb-2">
                      <span className="flex-shrink-0 w-6 h-6 rounded-full bg-primary/15 text-primary text-xs flex items-center justify-center font-medium">
                        {qa.question_index + 1}
                      </span>
                      <div className="flex-1">
                        <p className="text-sm font-medium">{qa.question_text}</p>
                      </div>
                    </div>
                    {qa.answer_text ? (
                      <div className="ml-8 mt-2">
                        <p className="text-sm text-muted-foreground bg-muted/50 p-3 rounded-md">
                          {qa.answer_text}
                        </p>
                        <div className="flex items-center gap-2 mt-2">
                          <div className="h-1.5 w-full bg-muted rounded-full overflow-hidden">
                            <div
                              className="h-full bg-green-500"
                              style={{ width: `${qa.confidence * 100}%` }}
                            />
                          </div>
                          <span className="text-xs text-muted-foreground whitespace-nowrap">
                            {Math.round(qa.confidence * 100)}%
                          </span>
                        </div>
                      </div>
                    ) : (
                      <p className="ml-8 text-sm text-yellow-600">Not answered</p>
                    )}
                  </div>
                ))}
              </CardContent>
            </Card>
          )}

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

          {/* IVR Menu Script */}
          {call.ivr_patterns && call.ivr_patterns.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <ListOrdered className="h-4 w-4" />
                  IVR Menu Script
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ol className="space-y-2">
                  {call.ivr_patterns.map((p, i) => (
                    <li key={i} className="flex items-start gap-2 text-sm">
                      <span className="flex-shrink-0 w-5 h-5 rounded-full bg-blue-500/15 text-blue-500 text-xs flex items-center justify-center font-medium">
                        {p.menu_level}
                      </span>
                      <div className="flex-1">
                        <p className="text-muted-foreground text-xs">Listen for: <span className="text-foreground">{p.detected_phrase}</span></p>
                        <Badge variant="outline" className="mt-1 text-xs">
                          {p.preferred_action === 'dtmf' ? `Press ${p.action_value}` : p.preferred_action === 'speech' ? `Say "${p.action_value}"` : 'Wait'}
                        </Badge>
                      </div>
                    </li>
                  ))}
                </ol>
              </CardContent>
            </Card>
          )}

          {/* IVR Navigation Log */}
          {call.events && call.events.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <Activity className="h-4 w-4" />
                  IVR Navigation Log
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3 max-h-[400px] overflow-y-auto">
                  {call.events.map((evt, i) => (
                    <div key={i} className="border-l-2 pl-3 py-1 text-sm border-muted-foreground/20">
                      <div className="flex items-center gap-2 mb-0.5">
                        <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                          {evt.event_type === 'ivr_menu_matched' ? 'Menu Matched' :
                           evt.event_type === 'ivr_credential_sent' ? 'Credential Sent' :
                           evt.event_type === 'human_detected' ? 'Human Detected' :
                           evt.event_type === 'ivr_system_speech' ? 'IVR Speech' :
                           evt.event_type}
                        </Badge>
                        {evt.action_taken && (
                          <span className="text-xs text-muted-foreground font-mono">{evt.action_taken}</span>
                        )}
                        {evt.timestamp && (
                          <LocalDateTime
                            value={evt.timestamp}
                            fallback=""
                            className="text-[10px] text-muted-foreground ml-auto"
                          />
                        )}
                      </div>
                      {evt.transcript && (
                        <p className="text-xs text-muted-foreground line-clamp-2">{evt.transcript}</p>
                      )}
                    </div>
                  ))}
                </div>
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
                    (msg: {
                      speaker: string
                      message: string
                      timestamp?: string
                      is_question?: boolean
                      is_answer?: boolean
                      related_qa_id?: string
                    }, i: number) => (
                      (() => {
                        const isAgentTranscript = msg.speaker === 'agent_transcript'
                        const transcriptParagraphs = isAgentTranscript
                          ? splitTranscriptParagraphs(msg.message)
                          : []

                        return (
                      <div
                        key={i}
                        className={`flex gap-3 ${
                          msg.speaker === 'agent' ? 'justify-end' : 'justify-start'
                        }`}
                      >
                        <div
                          className={`rounded-lg px-4 py-2 ${
                            isAgentTranscript
                              ? 'max-w-full border bg-sky-50/60'
                              : 'max-w-[80%]'
                          } ${
                            msg.speaker === 'agent'
                              ? 'bg-primary text-primary-foreground'
                              : 'bg-muted'
                          } ${
                            msg.is_question || msg.is_answer
                              ? 'ring-2 ring-yellow-400/50'
                              : ''
                          }`}
                        >
                          <div className="flex items-center gap-2 mb-1 flex-wrap">
                            <span className="text-xs font-medium opacity-80">
                              {msg.speaker === 'agent'
                                ? 'AI Agent'
                                : msg.speaker === 'ivr'
                                  ? 'IVR System'
                                  : msg.speaker === 'agent_transcript'
                                    ? 'Recorded Agent Call'
                                    : 'Representative'}
                            </span>
                            {msg.timestamp && (
                              <LocalDateTime
                                value={msg.timestamp}
                                fallback=""
                                className="text-xs opacity-60"
                              />
                            )}
                            {isAgentTranscript && (
                              <Badge variant="outline" className="text-xs h-5">
                                Auto Transcript
                              </Badge>
                            )}
                            {msg.is_question && (
                              <Badge variant="outline" className="text-xs h-5">
                                Q
                              </Badge>
                            )}
                            {msg.is_answer && (
                              <Badge variant="outline" className="text-xs h-5">
                                A
                              </Badge>
                            )}
                          </div>
                          {isAgentTranscript ? (
                            <div className="space-y-3">
                              {transcriptParagraphs.map((paragraph, paragraphIndex) => (
                                <p
                                  key={paragraphIndex}
                                  className="text-sm leading-6 text-foreground whitespace-pre-wrap"
                                >
                                  {paragraph}
                                </p>
                              ))}
                            </div>
                          ) : (
                            <p className="text-sm">{msg.message}</p>
                          )}
                        </div>
                      </div>
                        )
                      })()
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

          {/* Human Detection Feedback */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Brain className="h-4 w-4" />
                Human Detection Feedback
              </CardTitle>
            </CardHeader>
            <CardContent>
              {call.human_detection_correct !== null && call.human_detection_correct !== undefined && !feedbackResult ? (
                <div className="flex items-center gap-2 text-sm">
                  {call.human_detection_correct ? (
                    <>
                      <ThumbsUp className="h-4 w-4 text-green-500" />
                      <span className="text-green-500">Marked as correct</span>
                    </>
                  ) : (
                    <>
                      <ThumbsDown className="h-4 w-4 text-red-500" />
                      <span className="text-red-500">Marked as incorrect - system learned from this</span>
                    </>
                  )}
                </div>
              ) : feedbackResult ? (
                <div className="space-y-3">
                  <p className="text-sm text-muted-foreground">{feedbackResult.analysis}</p>
                  {feedbackResult.new_phrases.length > 0 ? (
                    <div>
                      <p className="text-sm font-medium flex items-center gap-1.5 mb-2">
                        <Sparkles className="h-3.5 w-3.5 text-yellow-500" />
                        {feedbackResult.new_phrases.length} new phrase{feedbackResult.new_phrases.length > 1 ? 's' : ''} learned:
                      </p>
                      <div className="space-y-1.5">
                        {feedbackResult.new_phrases.map((p, i) => (
                          <div key={i} className="flex items-center gap-2 text-sm">
                            <Badge variant="outline" className="text-[10px]">
                              {p.phrase_type === 'human' ? 'Human' :
                               p.phrase_type === 'ivr_definitive' ? 'IVR' :
                               p.phrase_type === 'ivr_passive' ? 'IVR Passive' : p.phrase_type}
                            </Badge>
                            <span className="font-mono text-xs">&quot;{p.phrase}&quot;</span>
                            <span className="text-xs text-muted-foreground ml-auto">{Math.round(p.confidence * 100)}%</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : (
                    <p className="text-sm text-green-500 flex items-center gap-1.5">
                      <ThumbsUp className="h-4 w-4" />
                      Feedback recorded. Thank you!
                    </p>
                  )}
                </div>
              ) : (
                <div className="space-y-3">
                  <p className="text-sm text-muted-foreground">
                    Was the human correctly detected in this call? Your feedback helps improve future calls.
                  </p>
                  <div className="flex gap-3">
                    <Button
                      size="sm"
                      variant="outline"
                      className="gap-1.5"
                      disabled={feedbackSubmitting}
                      onClick={() => handleFeedback(true)}
                    >
                      <ThumbsUp className="h-3.5 w-3.5" />
                      Yes, correct
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      className="gap-1.5 border-red-500/30 text-red-500 hover:bg-red-500/10"
                      disabled={feedbackSubmitting}
                      onClick={() => handleFeedback(false)}
                    >
                      {feedbackSubmitting ? (
                        <Loader2 className="h-3.5 w-3.5 animate-spin" />
                      ) : (
                        <ThumbsDown className="h-3.5 w-3.5" />
                      )}
                      No, incorrect
                    </Button>
                  </div>
                  {feedbackSubmitting && (
                    <p className="text-xs text-muted-foreground flex items-center gap-1.5">
                      <Loader2 className="h-3 w-3 animate-spin" />
                      Analyzing transcript for new detection phrases...
                    </p>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </motion.div>
  )
}
