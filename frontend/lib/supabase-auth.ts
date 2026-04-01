import type { AuthUser } from '@/types'

export interface StoredSession {
  access_token: string
  refresh_token: string
  token_type?: string
  expires_in?: number
  expires_at?: number
  user?: {
    id?: string
    email?: string
  }
}

const ACCESS_TOKEN_KEY = 'supabase_access_token'
const REFRESH_TOKEN_KEY = 'supabase_refresh_token'
const EXPIRES_AT_KEY = 'supabase_expires_at'

function getSupabaseUrl() {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL
  if (!url) throw new Error('NEXT_PUBLIC_SUPABASE_URL is not configured')
  return url.replace(/\/$/, '')
}

function getSupabaseAnonKey() {
  const key = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY
  if (!key) throw new Error('NEXT_PUBLIC_SUPABASE_ANON_KEY is not configured')
  return key
}

function authHeaders(accessToken?: string): HeadersInit {
  const headers: HeadersInit = {
    apikey: getSupabaseAnonKey(),
    'Content-Type': 'application/json',
  }
  if (accessToken) {
    headers.Authorization = `Bearer ${accessToken}`
  }
  return headers
}

export function buildAuthRedirectUrl(path: string) {
  const base =
    process.env.NEXT_PUBLIC_APP_URL ||
    (typeof window !== 'undefined' ? window.location.origin : '')

  if (!base) {
    throw new Error('NEXT_PUBLIC_APP_URL is not configured')
  }

  return `${base.replace(/\/$/, '')}${path}`
}

export function storeSession(session: StoredSession) {
  localStorage.setItem(ACCESS_TOKEN_KEY, session.access_token)
  localStorage.setItem(REFRESH_TOKEN_KEY, session.refresh_token)
  if (session.expires_at) {
    localStorage.setItem(EXPIRES_AT_KEY, String(session.expires_at))
  } else {
    localStorage.removeItem(EXPIRES_AT_KEY)
  }
}

export function getStoredSession(): StoredSession | null {
  if (typeof window === 'undefined') return null

  const access_token = localStorage.getItem(ACCESS_TOKEN_KEY)
  const refresh_token = localStorage.getItem(REFRESH_TOKEN_KEY)
  const expires_at = localStorage.getItem(EXPIRES_AT_KEY)
  if (!access_token || !refresh_token) return null

  return {
    access_token,
    refresh_token,
    expires_at: expires_at ? Number(expires_at) : undefined,
  }
}

export function clearStoredSession() {
  localStorage.removeItem(ACCESS_TOKEN_KEY)
  localStorage.removeItem(REFRESH_TOKEN_KEY)
  localStorage.removeItem(EXPIRES_AT_KEY)
}

export function parseSessionFromHash(hash: string): (StoredSession & { type?: string }) | null {
  if (!hash || !hash.includes('access_token=')) return null
  const params = new URLSearchParams(hash.replace(/^#/, ''))
  const access_token = params.get('access_token')
  const refresh_token = params.get('refresh_token')

  if (!access_token || !refresh_token) return null

  const expires_in = Number(params.get('expires_in') || 0) || undefined
  return {
    access_token,
    refresh_token,
    token_type: params.get('token_type') || undefined,
    expires_in,
    expires_at: expires_in ? Math.floor(Date.now() / 1000) + expires_in : undefined,
    type: params.get('type') || undefined,
  }
}

async function fetchJson<T>(input: RequestInfo, init: RequestInit): Promise<T> {
  const response = await fetch(input, init)
  const data = await response.json().catch(() => ({}))
  if (!response.ok) {
    const message = data.error_description || data.msg || data.error || 'Authentication request failed'
    throw new Error(message)
  }
  return data as T
}

export async function signInWithPassword(email: string, password: string): Promise<StoredSession> {
  const data = await fetchJson<StoredSession>(
    `${getSupabaseUrl()}/auth/v1/token?grant_type=password`,
    {
      method: 'POST',
      headers: authHeaders(),
      body: JSON.stringify({ email, password }),
    }
  )

  storeSession(data)
  return data
}

export async function signUpWithPassword(input: {
  email: string
  password: string
  username: string
}): Promise<{ user?: AuthUser | null; session?: StoredSession | null }> {
  const data = await fetchJson<{ user?: AuthUser | null; session?: StoredSession | null }>(
    `${getSupabaseUrl()}/auth/v1/signup`,
    {
      method: 'POST',
      headers: authHeaders(),
      body: JSON.stringify({
        email: input.email,
        password: input.password,
        data: { username: input.username },
        redirect_to: buildAuthRedirectUrl('/auth/callback'),
      }),
    }
  )

  return data
}

export async function signOut(accessToken?: string | null) {
  if (accessToken) {
    try {
      await fetch(`${getSupabaseUrl()}/auth/v1/logout`, {
        method: 'POST',
        headers: authHeaders(accessToken),
      })
    } catch {
      // Best-effort logout; local session is cleared below.
    }
  }
  clearStoredSession()
}

export async function refreshSession(refreshToken: string): Promise<StoredSession> {
  const data = await fetchJson<StoredSession>(
    `${getSupabaseUrl()}/auth/v1/token?grant_type=refresh_token`,
    {
      method: 'POST',
      headers: authHeaders(),
      body: JSON.stringify({ refresh_token: refreshToken }),
    }
  )

  storeSession(data)
  return data
}

export async function sendPasswordReset(email: string) {
  await fetchJson(
    `${getSupabaseUrl()}/auth/v1/recover`,
    {
      method: 'POST',
      headers: authHeaders(),
      body: JSON.stringify({
        email,
        redirect_to: buildAuthRedirectUrl('/reset-password'),
      }),
    }
  )
}

export async function updatePassword(accessToken: string, password: string) {
  await fetchJson(
    `${getSupabaseUrl()}/auth/v1/user`,
    {
      method: 'PUT',
      headers: authHeaders(accessToken),
      body: JSON.stringify({ password }),
    }
  )
}
