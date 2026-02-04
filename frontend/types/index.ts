// types/index.ts
// Type definitions for the Autonomous Credentialing System

export type CallState = 
  | 'initiating'
  | 'ivr_navigation'
  | 'on_hold'
  | 'speaking_with_human'
  | 'extracting_info'
  | 'completing'
  | 'failed';

export type CredentialingStatus =
  | 'initiated'
  | 'approved'
  | 'pending_review'
  | 'missing_documents'
  | 'denied'
  | 'office_closed'
  | 'failed';

export interface CredentialingRequest {
  id?: string;
  insurance_name: string;
  provider_name: string;
  npi: string;
  tax_id: string;
  address: string;
  insurance_phone: string;
  questions: string[];
  status?: CredentialingStatus;
  reference_number?: string;
  missing_documents?: string[];
  turnaround_days?: number;
  notes?: string;
  created_at?: string;
  completed_at?: string;
}

export interface CallStatus {
  success: boolean;
  call_id: string;
  call_sid?: string;
  call_state: CallState;
  insurance_name: string;
  provider_name: string;
  status?: CredentialingStatus;
  reference_number?: string;
  notes?: string;
}

export interface ConversationMessage {
  speaker: 'agent' | 'representative' | 'ivr';
  message: string;
  timestamp?: string;
}

export interface CallTranscript {
  success: boolean;
  call_id: string;
  conversation: ConversationMessage[];
  transcript: Array<{
    text: string;
    timestamp: string;
    confidence: number;
  }>;
}

export interface SystemMetrics {
  success: boolean;
  period_days: number;
  total_calls: number;
  approved: number;
  in_progress: number;
  success_rate: number;
}

export interface IVRKnowledge {
  id?: string;
  insurance_name: string;
  menu_level: number;
  detected_phrase: string;
  preferred_action: 'dtmf' | 'speech' | 'wait';
  action_value?: string;
  confidence_threshold?: number;
  success_rate?: number;
  attempts?: number;
}

export interface InsuranceProvider {
  id?: string;
  insurance_name: string;
  phone_number: string;
  department?: string;
  best_call_times?: {
    days: string[];
    hours: number[];
  };
  average_wait_time_minutes?: number;
  notes?: string;
  last_updated?: string;
}

export interface ScheduledFollowup {
  id: string;
  request_id: string;
  scheduled_date: string;
  action_type: 'retry_call' | 'follow_up_call' | 'submit_documents_then_call';
  status: 'pending' | 'completed' | 'failed';
  insurance_name: string;
  provider_name: string;
}

export interface APIResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

export interface DashboardStats {
  total_calls_today: number;
  active_calls: number;
  success_rate_7d: number;
  avg_duration_minutes: number;
  pending_followups: number;
}

export interface CallEvent {
  id: string;
  call_id: string;
  event_type: string;
  transcript?: string;
  action_taken?: string;
  confidence?: number;
  timestamp: string;
  metadata?: Record<string, any>;
}

export interface CallDetailEvent {
  event_type: string;
  transcript?: string;
  action_taken?: string;
  confidence?: number;
  timestamp?: string;
  metadata?: Record<string, any>;
}

export interface CallDetail {
  id: string;
  insurance_name: string;
  provider_name: string;
  npi: string;
  tax_id: string;
  address: string;
  insurance_phone: string;
  questions: string[];
  status?: CredentialingStatus;
  reference_number?: string;
  missing_documents?: string[];
  turnaround_days?: number;
  notes?: string;
  created_at?: string;
  updated_at?: string;
  completed_at?: string;
  conversation: ConversationMessage[];
  events: CallDetailEvent[];
}

export interface LangSmithTrace {
  run_id: string;
  name: string;
  run_type: string;
  start_time: string;
  end_time?: string;
  inputs?: Record<string, any>;
  outputs?: Record<string, any>;
  error?: string;
  latency_ms?: number;
  total_tokens?: number;
  prompt_tokens?: number;
  completion_tokens?: number;
  total_cost?: number;
}
