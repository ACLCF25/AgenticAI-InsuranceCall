// lib/api.ts
// API client for communicating with the Python backend

import axios, { AxiosInstance } from 'axios';
import type {
  CredentialingRequest,
  CallStatus,
  CallTranscript,
  SystemMetrics,
  IVRKnowledge,
  ScheduledFollowup,
  APIResponse,
  DashboardStats,
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

    // Add request interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        console.error('API Error:', error.response?.data || error.message);
        return Promise.reject(error);
      }
    );
  }

  // Health check
  async healthCheck(): Promise<APIResponse> {
    const { data } = await this.client.get('/health');
    return data;
  }

  // Start a new credentialing call
  async startCall(request: CredentialingRequest): Promise<APIResponse> {
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
    const { data } = await this.client.get(`/ivr-knowledge/${insuranceName}`);
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
}

export const api = new APIClient();
