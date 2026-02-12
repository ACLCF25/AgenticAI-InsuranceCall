'use client'

import { motion } from 'framer-motion'
import { InsuranceManagement } from '@/components/dashboard/insurance-management'

export default function InsurancePage() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      <InsuranceManagement />
    </motion.div>
  )
}
