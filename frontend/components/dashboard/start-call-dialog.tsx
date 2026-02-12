'use client'

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { StartCallForm } from './start-call-form'

interface StartCallDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function StartCallDialog({ open, onOpenChange }: StartCallDialogProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[90vh] max-w-2xl overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Start New Credentialing Call</DialogTitle>
          <DialogDescription>
            Enter the details for the credentialing call. The AI agent will handle the process autonomously.
          </DialogDescription>
        </DialogHeader>

        <StartCallForm
          isActive={open}
          onCancel={() => onOpenChange(false)}
          onSuccess={() => onOpenChange(false)}
        />
      </DialogContent>
    </Dialog>
  )
}
