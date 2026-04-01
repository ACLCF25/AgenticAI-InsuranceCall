// lib/api.ts
// API client for communicating with the Python backend

import axios, { AxiosInstance, InternalAxiosRequestConfig } from 'axios';
import type {
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
  TwilioNumber,
  HumanDetectionPhrase,
  HumanDetectionFeedbackResponse,
} from '@/types';

class APIClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000/api',
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Synchronously seed the auth token from localStorage so the header is
    // present on the very first React Query request (before any useEffect runs).
    if (typeof window !== 'undefined') {
      const token = localStorage.getItem('auth_token')
      if (token) {
        this.client.defaults.headers.common['Authorization'] = `Bearer ${token}`
      }
    }

    // Response interceptor – attempt token refresh on 401 (expired token)
    this.client.interceptors.response.use(
      (response) => response,
      async (error) => {
        const original = error.config as InternalAxiosRequestConfig & { _retry?: boolean }
        const status = error.response?.status
        const jwtErrorMessage = String(error.response?.data?.msg || '').toLowerCase()
        const isJwt422 =
          status === 422 &&
          (jwtErrorMessage.includes('token') ||
            jwtErrorMessage.includes('subject') ||
            jwtErrorMessage.includes('jwt'))

        if ((status === 401 || isJwt422) && !original._retry) {
          original._retry = true
          try {
            const refreshToken = typeof window !== 'undefined'
              ? localStorage.getItem('auth_refresh_token')
              : null
            if (!refreshToken) throw new Error('No refresh token')
            const res = await axios.post(
              `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000/api'}/auth/refresh`,
              {},
              { headers: { Authorization: `Bearer ${refreshToken}` } }
            )
            const newToken = res.data.access_token
            this.setAuthToken(newToken)
            return this.client(original)
          } catch {
            // Refresh failed – clear auth state and redirect to login
            if (typeof window !== 'undefined') {
              localStorage.removeItem('auth_token')
              localStorage.removeItem('auth_refresh_token')
              localStorage.removeItem('auth_user')
              window.location.href = '/login'
            }
          }
        }
        return Promise.reject(error)
      }
    )
  }

  // Set (or clear) the Authorization header on all future requests
  setAuthToken(token: string | null) {
    if (token) {
      this.client.defaults.headers.common['Authorization'] = `Bearer ${token}`
    } else {
      delete this.client.defaults.headers.common['Authorization']
    }
  }

  // Health check
  async healthCheck(): Promise<APIResponse> {
    const { data } = await this.client.get('/health');
    return data;
  }

  // Start a new credentialing call
  async startCall(request: CredentialingRequest): Promise<StartCallResponse> {
    const { data } = await this.client.post('/start-call', request);
    return data;
  }

  // Get call status
  async getCallStatus(callId: string): Promise<CallStatus> {
    const { data } = await this.client.get(`/call-status/${callId}`);
    return data;
  }

  // Get call transcript
  async getCallTranscript(callId: string): Promise<CallTranscript> {
    const { data } = await this.client.get(`/call-transcript/${callId}`);
    return data;
  }

  // Get full call detail from database
  async getCallDetail(callId: string): Promise<APIResponse<CallDetail>> {
    const { data } = await this.client.get(`/call-detail/${callId}`);
    return data;
  }

  // Get system metrics
  async getMetrics(): Promise<SystemMetrics> {
    const { data } = await this.client.get('/metrics');
    return data;
  }

  // Get scheduled follow-ups
  async getScheduledFollowups(): Promise<APIResponse<ScheduledFollowup[]>> {
    const { data } = await this.client.get('/scheduled-followups');
    return data;
  }

  // Add IVR knowledge
  async addIVRKnowledge(knowledge: IVRKnowledge): Promise<APIResponse> {
    const { data } = await this.client.post('/ivr-knowledge', knowledge);
    return data;
  }

  // Get IVR knowledge for an insurance provider
  async getIVRKnowledge(insuranceName: string): Promise<APIResponse<IVRKnowledge[]>> {
    const encodedInsuranceName = encodeURIComponent(insuranceName);
    const { data } = await this.client.get(`/ivr-knowledge/${encodedInsuranceName}`);
    return data;
  }

  // Get recent calls (custom endpoint - would need to add to backend)
  async getRecentCalls(limit: number = 20): Promise<APIResponse<CredentialingRequest[]>> {
    const { data } = await this.client.get(`/calls?limit=${limit}`);
    return data;
  }

  // Get dashboard stats (custom endpoint - would need to add to backend)
  async getDashboardStats(): Promise<APIResponse<DashboardStats>> {
    const { data } = await this.client.get('/dashboard-stats');
    return data;
  }

  // Cancel/stop a call
  async cancelCall(callId: string): Promise<APIResponse> {
    const { data } = await this.client.post(`/call/${callId}/cancel`);
    return data;
  }

  // Execute a scheduled follow-up
  async executeFollowup(followupId: string): Promise<APIResponse> {
    const { data } = await this.client.post(`/followup/${followupId}/execute`);
    return data;
  }

  // Get all insurance providers
  async getInsuranceProviders(): Promise<APIResponse<InsuranceProvider[]>> {
    const { data } = await this.client.get('/insurance-providers');
    return data;
  }

  // Add a new insurance provider
  async addInsuranceProvider(provider: Omit<InsuranceProvider, 'id' | 'last_updated'>): Promise<APIResponse> {
    const { data } = await this.client.post('/insurance-providers', provider);
    return data;
  }

  // Update an insurance provider
  async updateInsuranceProvider(id: string, provider: Omit<InsuranceProvider, 'id' | 'last_updated'>): Promise<APIResponse> {
    const { data } = await this.client.put(`/insurance-providers/${id}`, provider);
    return data;
  }

  // Delete an insurance provider
  async deleteInsuranceProvider(id: string): Promise<APIResponse> {
    const { data } = await this.client.delete(`/insurance-providers/${id}`);
    return data;
  }

  // Delete IVR knowledge entry
  async deleteIVRKnowledge(id: string): Promise<APIResponse> {
    const { data } = await this.client.delete(`/ivr-knowledge/${id}`);
    return data;
  }

  // Update IVR knowledge entry
  async updateIVRKnowledge(id: string, knowledge: Omit<IVRKnowledge, 'id' | 'insurance_name'>): Promise<APIResponse> {
    const { data } = await this.client.put(`/ivr-knowledge/${id}`, knowledge);
    return data;
  }

  // Fetch call recording as a blob (carries the auth header that a plain <audio src> cannot)
  async getCallRecordingBlob(callId: string): Promise<Blob> {
    const { data } = await this.client.get(`/call-recording/${callId}/stream`, {
      responseType: 'blob',
    });
    return data;
  }

  // Transfer an active AI call to a real human agent
  async transferToAgent(callId: string, agentPhone: string): Promise<APIResponse> {
    const { data } = await this.client.post('/transfer-to-agent', {
      call_id: callId,
      agent_phone: agentPhone,
    });
    return data;
  }

  // ---- User Management (admin) ----

  async getUsers(): Promise<{ users: AdminUser[] }> {
    const { data } = await this.client.get('/auth/users');
    return data;
  }

  async createUser(user: { username: string; email: string; password: string; role: string }): Promise<{ user: AdminUser }> {
    const { data } = await this.client.post('/auth/register', user);
    return data;
  }

  async updateUser(userId: string, updates: { is_active?: boolean; role?: string }): Promise<{ user: AdminUser }> {
    const { data } = await this.client.patch(`/auth/users/${userId}`, updates);
    return data;
  }

  // ---- Twilio Number Management (admin) ----

  async getTwilioNumbers(): Promise<{ numbers: TwilioNumber[] }> {
    const { data } = await this.client.get('/twilio-numbers');
    return data;
  }

  async addTwilioNumber(number: { phone_number: string; friendly_name?: string }): Promise<{ number: TwilioNumber }> {
    const { data } = await this.client.post('/twilio-numbers', number);
    return data;
  }

  async updateTwilioNumber(id: string, updates: { is_active: boolean }): Promise<{ number: TwilioNumber }> {
    const { data } = await this.client.patch(`/twilio-numbers/${id}`, updates);
    return data;
  }

  async deleteTwilioNumber(id: string): Promise<{ message: string }> {
    const { data } = await this.client.delete(`/twilio-numbers/${id}`);
    return data;
  }

  // ---- Human Detection Self-Learning ----

  async submitHumanDetectionFeedback(callId: string, correct: boolean): Promise<HumanDetectionFeedbackResponse> {
    const { data } = await this.client.post(`/call/${callId}/human-detection-feedback`, { correct });
    return data;
  }

  async getHumanDetectionPhrases(insuranceName?: string): Promise<{ success: boolean; phrases: HumanDetectionPhrase[] }> {
    const params = insuranceName ? `?insurance_name=${encodeURIComponent(insuranceName)}` : '';
    const { data } = await this.client.get(`/human-detection-phrases${params}`);
    return data;
  }

  async addHumanDetectionPhrase(phrase: string, phraseType: string, insuranceName?: string): Promise<APIResponse> {
    const { data } = await this.client.post('/human-detection-phrases', {
      phrase, phrase_type: phraseType, insurance_name: insuranceName || null,
    });
    return data;
  }

  async deleteHumanDetectionPhrase(id: string): Promise<APIResponse> {
    const { data } = await this.client.delete(`/human-detection-phrases/${id}`);
    return data;
  }

  async updateHumanDetectionPhrase(id: string, updates: Partial<HumanDetectionPhrase>): Promise<APIResponse> {
    const { data } = await this.client.patch(`/human-detection-phrases/${id}`, updates);
    return data;
  }
}

export const api = new APIClient();
