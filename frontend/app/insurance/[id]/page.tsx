'use client'

import { motion } from 'framer-motion'
import { ArrowLeft, Building2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import Link from 'next/link'

export default function ProviderDetailPage() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      <Button asChild variant="ghost" size="sm" className="gap-1.5">
        <Link href="/insurance">
          <ArrowLeft className="h-4 w-4" />
          Back to Providers
        </Link>
      </Button>

      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Building2 className="h-5 w-5 text-muted-foreground" />
            <div>
              <CardTitle>Provider Detail</CardTitle>
              <CardDescription>Provider details and IVR knowledge</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground py-8 text-center">
            Provider detail page — will show provider info, IVR knowledge, and call history.
          </p>
        </CardContent>
      </Card>
    </motion.div>
  )
}
