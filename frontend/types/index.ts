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
  | 'failed'
  | 'transferred';

export type UserRole = 'super_admin' | 'admin' | 'agent';
export type ApprovalStatus = 'pending' | 'approved' | 'rejected';

export interface CredentialingRequest {
  id?: string;
  insurance_name: string;
  provider_name: string;
  npi: string;
  tax_id: string;
  address: string;
  insurance_phone: string;
  provider_phone?: string;
  questions: string[];
  status?: CredentialingStatus;
  reference_number?: string;
  missing_documents?: string[];
  turnaround_days?: number;
  notes?: string;
  created_at?: string;
  completed_at?: string;
  call_mode?: string;
  agent_phone?: string;
  initiated_by?: string | null;
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
  ivr_asks_npi?: boolean;
  ivr_npi_method?: 'speech' | 'dtmf';
  ivr_asks_tax_id?: boolean;
  ivr_tax_id_method?: 'speech' | 'dtmf';
  ivr_tax_id_digits_to_send?: number;
  ivr_npi_suffix?: string | null;       // '*', '#', or null
  ivr_tax_id_suffix?: string | null;    // '*', '#', or null
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

export interface CallLogEntry {
  id: number;
  call_id: string;
  call_sid?: string | null;
  logged_at: string | null;
  level: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR' | 'CRITICAL' | string;
  logger?: string | null;
  function?: string | null;
  line?: number | null;
  message: string;
}

export interface StartCallResponse extends APIResponse {
  call_id?: string;
  request_id?: string | null;
  provider?: string;
  insurance?: string;
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

export interface QAPair {
  id: string;
  question_index: number;
  question_text: string;
  answer_text?: string;
  confidence: number;
}

export interface CallDetail {
  id: string;
  insurance_name: string;
  provider_name: string;
  npi: string;
  tax_id: string;
  address: string;
  insurance_phone: string;
  provider_phone?: string;
  questions: string[];
  status?: CredentialingStatus;
  reference_number?: string;
  missing_documents?: string[];
  turnaround_days?: number;
  notes?: string;
  created_at?: string;
  updated_at?: string;
  completed_at?: string;
  transfer_started_at?: string | null;
  call_mode?: string;
  agent_phone?: string;
  initiated_by?: string | null;
  conversation: ConversationMessage[];
  events: CallDetailEvent[];
  ivr_patterns?: Array<{
    menu_level: number;
    detected_phrase: string;
    preferred_action: string;
    action_value: string;
  }>;
  recording?: {
    available: boolean;
    url: string;
    duration?: number;
    recording_type?: 'ai' | 'agent' | 'both';
    status?: 'completed' | 'failed' | 'pending' | 'processing';
    created_at?: string;
  };
  qa_pairs?: QAPair[];
  human_detection_correct?: boolean | null;
  metrics?: {
    duration_seconds?: number;
    ivr_navigation_time_seconds?: number;
    hold_time_seconds?: number;
    human_interaction_time_seconds?: number;
    successful?: boolean;
  };
}

export interface HumanDetectionPhrase {
  id: string;
  phrase: string;
  phrase_type: 'human' | 'ivr_definitive' | 'ivr_passive' | 'simple_greeting';
  insurance_name: string | null;
  source: 'manual' | 'auto_review' | 'feedback';
  confidence: number;
  times_seen: number;
  times_correct: number;
  is_active: boolean;
  source_call_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface HumanDetectionFeedbackResponse {
  success: boolean;
  correct: boolean;
  new_phrases: Array<{
    phrase: string;
    phrase_type: string;
    confidence: number;
  }>;
  analysis?: string;
  analysis_error?: string;
}

export interface AdminUser {
  id: string;
  username: string | null;
  email: string;
  role: UserRole | null;
  approval_status: ApprovalStatus;
  email_confirmed: boolean;
  approved_by?: string | null;
  approved_by_username?: string | null;
  approved_at?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
  profile_missing?: boolean;
}

export interface AuthUser extends AdminUser {}

export interface AuditLogEntry {
  id: string;
  user_id?: string | null;
  action: string;
  resource_type?: string | null;
  resource_id?: string | null;
  details?: Record<string, any> | null;
  ip_address?: string | null;
  timestamp: string;
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
