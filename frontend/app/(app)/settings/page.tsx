'use client'

import { motion } from 'framer-motion'
import { Settings, Globe, Server, Bell } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'

export default function SettingsPage() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="max-w-2xl space-y-6"
    >
      {/* API Configuration */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Globe className="h-4 w-4 text-muted-foreground" />
            <div>
              <CardTitle className="text-base">API Configuration</CardTitle>
              <CardDescription>Backend connection settings</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex items-center justify-between rounded-lg border p-3">
            <div>
              <p className="text-sm font-medium">API URL</p>
              <p className="text-xs text-muted-foreground font-mono">
                {process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000/api'}
              </p>
            </div>
            <div className="h-2 w-2 rounded-full bg-green-500 pulse-dot" />
          </div>
          <div className="flex items-center justify-between rounded-lg border p-3">
            <div>
              <p className="text-sm font-medium">WebSocket URL</p>
              <p className="text-xs text-muted-foreground font-mono">
                {process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:5000'}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* System Info */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Server className="h-4 w-4 text-muted-foreground" />
            <div>
              <CardTitle className="text-base">System Information</CardTitle>
              <CardDescription>Application details</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid gap-3 text-sm">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Version</span>
              <span className="font-mono">1.0.0</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Framework</span>
              <span className="font-mono">Next.js 16</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Backend</span>
              <span className="font-mono">Flask + LangGraph</span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Notifications placeholder */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Bell className="h-4 w-4 text-muted-foreground" />
            <div>
              <CardTitle className="text-base">Notifications</CardTitle>
              <CardDescription>Configure alert preferences</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground py-4 text-center">
            Notification settings coming soon
          </p>
        </CardContent>
      </Card>
    </motion.div>
  )
}
