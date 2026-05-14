// lib/api.ts
// API client for communicating with the Python backend

import axios, { AxiosInstance, InternalAxiosRequestConfig } from 'axios'
import type {
  AuditLogEntry,
  CallLogEntry,
  CredentialingRequest,
  CallStatus,
  CallTranscript,
  CallDetail,
  SystemMetrics,
  IVRKnowledge,
  InsuranceProvider,
  ScheduledFollowup,
  APIResponse,
  StartCallResponse,
  DashboardStats,
  AdminUser,
  AuthUser,
  HumanDetectionPhrase,
  HumanDetectionFeedbackResponse,
} from '@/types'
import { clearStoredSession, getStoredSession, refreshSession } from './supabase-auth'

class APIClient {
  private client: AxiosInstance

  constructor() {
    this.client = axios.create({
      baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000/api',
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    })

    if (typeof window !== 'undefined') {
      const session = getStoredSession()
      if (session?.access_token) {
        this.client.defaults.headers.common['Authorization'] = `Bearer ${session.access_token}`
      }
    }

    this.client.interceptors.response.use(
      (response) => response,
      async (error) => {
        const original = error.config as InternalAxiosRequestConfig & { _retry?: boolean }
        const status = error.response?.status

        if (status === 401 && !original._retry) {
          original._retry = true
          try {
            const session = getStoredSession()
            if (!session?.refresh_token) throw new Error('No refresh token')
            const refreshed = await refreshSession(session.refresh_token)
            this.setAuthToken(refreshed.access_token)
            return this.client(original)
          } catch {
            if (typeof window !== 'undefined') {
              clearStoredSession()
              window.location.href = '/login'
            }
          }
        }

        return Promise.reject(error)
      }
    )
  }

  setAuthToken(token: string | null) {
    if (token) {
      this.client.defaults.headers.common['Authorization'] = `Bearer ${token}`
    } else {
      delete this.client.defaults.headers.common['Authorization']
    }
  }

  async healthCheck(): Promise<APIResponse> {
    const { data } = await this.client.get('/health')
    return data
  }

  async startCall(request: CredentialingRequest): Promise<StartCallResponse> {
    const { data } = await this.client.post('/start-call', request)
    return data
  }

  async getCallStatus(callId: string): Promise<CallStatus> {
    const { data } = await this.client.get(`/call-status/${callId}`)
    return data
  }

  async getCallTranscript(callId: string): Promise<CallTranscript> {
    const { data } = await this.client.get(`/call-transcript/${callId}`)
    return data
  }

  async getCallDetail(callId: string): Promise<APIResponse<CallDetail>> {
    const { data } = await this.client.get(`/call-detail/${callId}`)
    return data
  }

  async getCallLogs(
    callId: string,
    opts: { level?: string; limit?: number } = {}
  ): Promise<APIResponse<CallLogEntry[]> & { count?: number }> {
    const params = new URLSearchParams()
    if (opts.level) params.set('level', opts.level)
    if (opts.limit) params.set('limit', String(opts.limit))
    const qs = params.toString()
    const url = `/call-logs/${callId}${qs ? `?${qs}` : ''}`
    const { data } = await this.client.get(url)
    // Backend returns { success, call_id, count, logs: [...] }.
    // Normalize to APIResponse shape so callers can use `data` like other endpoints.
    return { ...data, data: data.logs }
  }

  async getMetrics(): Promise<SystemMetrics> {
    const { data } = await this.client.get('/metrics')
    return data
  }

  async getScheduledFollowups(): Promise<APIResponse<ScheduledFollowup[]>> {
    const { data } = await this.client.get('/scheduled-followups')
    return data
  }

  async addIVRKnowledge(knowledge: IVRKnowledge): Promise<APIResponse> {
    const { data } = await this.client.post('/ivr-knowledge', knowledge)
    return data
  }

  async getIVRKnowledge(insuranceName: string): Promise<APIResponse<IVRKnowledge[]>> {
    const encodedInsuranceName = encodeURIComponent(insuranceName)
    const { data } = await this.client.get(`/ivr-knowledge/${encodedInsuranceName}`)
    return data
  }

  async getRecentCalls(limit = 20): Promise<APIResponse<CredentialingRequest[]>> {
    const { data } = await this.client.get(`/calls?limit=${limit}`)
    return data
  }

  async getDashboardStats(): Promise<APIResponse<DashboardStats>> {
    const { data } = await this.client.get('/dashboard-stats')
    return data
  }

  async cancelCall(callId: string): Promise<APIResponse> {
    const { data } = await this.client.post(`/call/${callId}/cancel`)
    return data
  }

  async executeFollowup(followupId: string): Promise<APIResponse> {
    const { data } = await this.client.post(`/followup/${followupId}/execute`)
    return data
  }

  async getInsuranceProviders(): Promise<APIResponse<InsuranceProvider[]>> {
    const { data } = await this.client.get('/insurance-providers')
    return data
  }

  async addInsuranceProvider(provider: Omit<InsuranceProvider, 'id' | 'last_updated'>): Promise<APIResponse> {
    const { data } = await this.client.post('/insurance-providers', provider)
    return data
  }

  async updateInsuranceProvider(id: string, provider: Omit<InsuranceProvider, 'id' | 'last_updated'>): Promise<APIResponse> {
    const { data } = await this.client.put(`/insurance-providers/${id}`, provider)
    return data
  }

  async deleteInsuranceProvider(id: string): Promise<APIResponse> {
    const { data } = await this.client.delete(`/insurance-providers/${id}`)
    return data
  }

  async deleteIVRKnowledge(id: string): Promise<APIResponse> {
    const { data } = await this.client.delete(`/ivr-knowledge/${id}`)
    return data
  }

  async updateIVRKnowledge(id: string, knowledge: Omit<IVRKnowledge, 'id' | 'insurance_name'>): Promise<APIResponse> {
    const { data } = await this.client.put(`/ivr-knowledge/${id}`, knowledge)
    return data
  }

  async getCallRecordingBlob(callId: string): Promise<Blob> {
    const { data } = await this.client.get(`/call-recording/${callId}/stream`, {
      responseType: 'blob',
    })
    return data
  }

  async transferToAgent(callId: string, agentPhone: string): Promise<APIResponse> {
    const { data } = await this.client.post('/transfer-to-agent', {
      call_id: callId,
      agent_phone: agentPhone,
    })
    return data
  }

  async getCurrentUser(): Promise<{ user: AuthUser }> {
    const { data } = await this.client.get('/auth/me')
    return data
  }

  async getUsers(): Promise<{ users: AdminUser[] }> {
    const { data } = await this.client.get('/auth/users')
    return data
  }

  async updateUser(userId: string, updates: { approval_status?: string; role?: string }): Promise<{ user: AdminUser }> {
    const { data } = await this.client.patch(`/auth/users/${userId}`, updates)
    return data
  }

  async getAuditLogs(): Promise<{ success: boolean; data: AuditLogEntry[] }> {
    const { data } = await this.client.get('/audit-logs?limit=100')
    return data
  }

  async submitHumanDetectionFeedback(callId: string, correct: boolean): Promise<HumanDetectionFeedbackResponse> {
    const { data } = await this.client.post(`/call/${callId}/human-detection-feedback`, { correct })
    return data
  }

  async getHumanDetectionPhrases(insuranceName?: string): Promise<{ success: boolean; phrases: HumanDetectionPhrase[] }> {
    const params = insuranceName ? `?insurance_name=${encodeURIComponent(insuranceName)}` : ''
    const { data } = await this.client.get(`/human-detection-phrases${params}`)
    return data
  }

  async addHumanDetectionPhrase(phrase: string, phraseType: string, insuranceName?: string): Promise<APIResponse> {
    const { data } = await this.client.post('/human-detection-phrases', {
      phrase,
      phrase_type: phraseType,
      insurance_name: insuranceName || null,
    })
    return data
  }

  async deleteHumanDetectionPhrase(id: string): Promise<APIResponse> {
    const { data } = await this.client.delete(`/human-detection-phrases/${id}`)
    return data
  }

  async updateHumanDetectionPhrase(id: string, updates: Partial<HumanDetectionPhrase>): Promise<APIResponse> {
    const { data } = await this.client.patch(`/human-detection-phrases/${id}`, updates)
    return data
  }
}

export const api = new APIClient()
