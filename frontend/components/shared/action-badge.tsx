import { Badge } from '@/components/ui/badge'

interface ActionBadgeProps {
  action: string
  size?: 'default' | 'sm'
}

export function ActionBadge({ action, size = 'default' }: ActionBadgeProps) {
  const className = size === 'sm' ? 'text-xs' : ''
  switch (action) {
    case 'dtmf':
      return <Badge variant="secondary" className={className}>Press Button</Badge>
    case 'speech':
      return <Badge variant="outline" className={className}>Say Phrase</Badge>
    case 'wait':
      return <Badge className={className}>Wait</Badge>
    default:
      return <Badge variant="secondary" className={className}>{action}</Badge>
  }
}
