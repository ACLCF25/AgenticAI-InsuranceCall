'use client'

import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { motion } from 'framer-motion'
import { ArrowLeft } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { StartCallForm } from '@/components/dashboard/start-call-form'

export default function NewCallPage() {
  const router = useRouter()

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="mx-auto max-w-3xl space-y-6"
    >
      <Button asChild variant="ghost" size="sm" className="gap-1.5">
        <Link href="/calls">
          <ArrowLeft className="h-4 w-4" />
          Back to Calls
        </Link>
      </Button>

      <Card>
        <CardHeader>
          <CardTitle>Start New Credentialing Call</CardTitle>
          <CardDescription>
            Fill in the details below to initiate an AI-powered credentialing call.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <StartCallForm
            isActive
            onCancel={() => router.push('/calls')}
            onSuccess={(response) => {
              if (response.request_id) {
                router.push(`/calls/${response.request_id}`)
                return
              }
              if (response.call_id) {
                router.push(`/calls/${response.call_id}`)
                return
              }
              router.push('/calls')
            }}
          />
        </CardContent>
      </Card>
    </motion.div>
  )
}
