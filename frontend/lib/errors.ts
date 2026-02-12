import { AxiosError } from 'axios'
import { toast } from 'sonner'

export interface ApiErrorResponse {
  error?: string
  message?: string
  detail?: string
}

export function getErrorMessage(error: unknown): string {
  if (error instanceof AxiosError) {
    const data = error.response?.data as ApiErrorResponse | undefined
    return data?.error || data?.message || data?.detail || error.message
  }
  if (error instanceof Error) {
    return error.message
  }
  return 'An unexpected error occurred'
}

export function handleMutationError(title: string) {
  return (error: unknown) => {
    toast.error(title, { description: getErrorMessage(error) })
  }
}
