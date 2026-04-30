// lib/utils.ts
// Utility functions for the application

import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"
import { formatDistanceToNow } from 'date-fns';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

function parseDateValue(date: string | Date): Date | null {
  const parsed = date instanceof Date ? date : new Date(date)
  return Number.isNaN(parsed.getTime()) ? null : parsed
}

export function formatDate(date: string | Date): string {
  const parsed = parseDateValue(date)
  if (!parsed) return ''

  return new Intl.DateTimeFormat(undefined, {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  }).format(parsed)
}

export function formatRelativeTime(date: string | Date): string {
  const parsed = parseDateValue(date)
  if (!parsed) return ''

  return formatDistanceToNow(parsed, { addSuffix: true });
}

export function formatDuration(seconds: number): string {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = seconds % 60;

  if (hours > 0) {
    return `${hours}h ${minutes}m ${secs}s`;
  } else if (minutes > 0) {
    return `${minutes}m ${secs}s`;
  } else {
    return `${secs}s`;
  }
}

export function formatPhoneNumber(phone?: string | null): string {
  // Gracefully handle missing values
  if (!phone) return '';

  // Format: +1 (234) 567-8900
  const cleaned = phone.replace(/\D/g, '');
  const match = cleaned.match(/^1?(\d{3})(\d{3})(\d{4})$/);
  
  if (match) {
    return `+1 (${match[1]}) ${match[2]}-${match[3]}`;
  }
  
  return phone;
}

export function normalizeUsPhone(phone?: string | null): string | null {
  const digits = (phone || '').replace(/\D/g, '')
  if (digits.length === 10) return `+1${digits}`
  if (digits.length === 11 && digits.startsWith('1')) return `+${digits}`
  return null
}

export function formatUsPhoneInput(phone?: string | null): string {
  const value = phone || ''
  const normalized = normalizeUsPhone(value)

  if (normalized) {
    return formatPhoneNumber(normalized)
  }

  return value.replace(/[^\d+\-().\s]/g, '')
}

export function formatStatus(status: string): string {
  return status.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
}

export function getStatusColor(status: string): string {
  const colors: Record<string, string> = {
    'initiated': 'bg-blue-500/15 text-blue-400 border border-blue-500/20',
    'approved': 'bg-green-500/15 text-green-400 border border-green-500/20',
    'pending_review': 'bg-yellow-500/15 text-yellow-400 border border-yellow-500/20',
    'missing_documents': 'bg-orange-500/15 text-orange-400 border border-orange-500/20',
    'denied': 'bg-red-500/15 text-red-400 border border-red-500/20',
    'office_closed': 'bg-gray-500/15 text-gray-400 border border-gray-500/20',
    'failed': 'bg-red-500/15 text-red-400 border border-red-500/20',
  };

  return colors[status] || 'bg-gray-500/15 text-gray-400 border border-gray-500/20';
}

export function getCallStateColor(state: string): string {
  const colors: Record<string, string> = {
    'initiating': 'bg-blue-500/15 text-blue-400 border border-blue-500/20',
    'ivr_navigation': 'bg-purple-500/15 text-purple-400 border border-purple-500/20',
    'on_hold': 'bg-yellow-500/15 text-yellow-400 border border-yellow-500/20',
    'speaking_with_human': 'bg-green-500/15 text-green-400 border border-green-500/20',
    'extracting_info': 'bg-indigo-500/15 text-indigo-400 border border-indigo-500/20',
    'completing': 'bg-teal-500/15 text-teal-400 border border-teal-500/20',
    'failed': 'bg-red-500/15 text-red-400 border border-red-500/20',
  };

  return colors[state] || 'bg-gray-500/15 text-gray-400 border border-gray-500/20';
}

export function formatCurrency(amount: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
  }).format(amount);
}

export function calculateSuccessRate(approved: number, total: number): number {
  if (total === 0) return 0;
  return Math.round((approved / total) * 100);
}

export function downloadJSON(data: any, filename: string): void {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout;
  
  return function executedFunction(...args: Parameters<T>) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

export function truncate(str: string, length: number): string {
  if (str.length <= length) return str;
  return str.substring(0, length) + '...';
}

export function capitalizeFirst(str: string): string {
  return str.charAt(0).toUpperCase() + str.slice(1);
}

export function parseNPI(npi: string): string {
  // Format NPI as XXX-XXX-XXXX
  const cleaned = npi.replace(/\D/g, '');
  const match = cleaned.match(/^(\d{3})(\d{3})(\d{4})$/);
  
  if (match) {
    return `${match[1]}-${match[2]}-${match[3]}`;
  }
  
  return npi;
}
