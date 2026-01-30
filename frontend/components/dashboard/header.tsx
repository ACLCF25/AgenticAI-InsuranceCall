// components/dashboard/header.tsx
'use client'

import { Phone, Plus } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { StartCallDialog } from './start-call-dialog'
import { useState } from 'react'

export function DashboardHeader() {
  const [dialogOpen, setDialogOpen] = useState(false)

  return (
    <header className="border-b bg-card">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary text-primary-foreground">
              <Phone className="h-6 w-6" />
            </div>
            <div>
              <h1 className="text-2xl font-bold">Credentialing Agent</h1>
              <p className="text-sm text-muted-foreground">
                Autonomous AI Insurance Credentialing System
              </p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <Button
              onClick={() => {
                console.log('Button clicked, opening dialog')
                setDialogOpen(true)
              }}
              className="gap-2"
            >
              <Plus className="h-4 w-4" />
              Start New Call
            </Button>
          </div>
        </div>
      </div>

      <StartCallDialog 
        open={dialogOpen} 
        onOpenChange={setDialogOpen} 
      />
    </header>
  )
}
