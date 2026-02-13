import { Skeleton } from '@/components/ui/skeleton'

export default function InsuranceLoading() {
  return (
    <div className="space-y-6">
      <Skeleton className="h-10 w-48 rounded-lg" />
      <Skeleton className="h-[400px] rounded-lg" />
    </div>
  )
}
