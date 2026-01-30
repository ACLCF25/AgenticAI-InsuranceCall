// lib/utils.ts
// Utility functions for the application

import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"
import { format, formatDistanceToNow } from 'date-fns';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatDate(date: string | Date): string {
  return format(new Date(date), 'MMM d, yyyy HH:mm');
}

export function formatRelativeTime(date: string | Date): string {
  return formatDistanceToNow(new Date(date), { addSuffix: true });
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

export function formatPhoneNumber(phone: string): string {
  // Format: +1 (234) 567-8900
  const cleaned = phone.replace(/\D/g, '');
  const match = cleaned.match(/^1?(\d{3})(\d{3})(\d{4})$/);
  
  if (match) {
    return `+1 (${match[1]}) ${match[2]}-${match[3]}`;
  }
  
  return phone;
}

export function getStatusColor(status: string): string {
  const colors: Record<string, string> = {
    'initiated': 'bg-blue-100 text-blue-800',
    'approved': 'bg-green-100 text-green-800',
    'pending_review': 'bg-yellow-100 text-yellow-800',
    'missing_documents': 'bg-orange-100 text-orange-800',
    'denied': 'bg-red-100 text-red-800',
    'office_closed': 'bg-gray-100 text-gray-800',
    'failed': 'bg-red-100 text-red-800',
  };
  
  return colors[status] || 'bg-gray-100 text-gray-800';
}

export function getCallStateColor(state: string): string {
  const colors: Record<string, string> = {
    'initiating': 'bg-blue-100 text-blue-800',
    'ivr_navigation': 'bg-purple-100 text-purple-800',
    'on_hold': 'bg-yellow-100 text-yellow-800',
    'speaking_with_human': 'bg-green-100 text-green-800',
    'extracting_info': 'bg-indigo-100 text-indigo-800',
    'completing': 'bg-teal-100 text-teal-800',
    'failed': 'bg-red-100 text-red-800',
  };
  
  return colors[state] || 'bg-gray-100 text-gray-800';
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
