import { Skeleton } from '@/components/ui/skeleton'

export default function AnalyticsLoading() {
  return (
    <div className="space-y-6">
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {[...Array(4)].map((_, i) => (
          <Skeleton key={i} className="h-24 rounded-lg" />
        ))}
      </div>
      <div className="grid gap-6 lg:grid-cols-2">
        <Skeleton className="h-80 rounded-lg" />
        <Skeleton className="h-80 rounded-lg" />
      </div>
    </div>
  )
}
