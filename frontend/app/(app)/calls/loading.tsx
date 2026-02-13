import { Skeleton } from '@/components/ui/skeleton'

export default function CallsLoading() {
  return (
    <div className="space-y-6">
      <Skeleton className="h-10 w-64 rounded-lg" />
      <Skeleton className="h-[500px] rounded-lg" />
    </div>
  )
}
