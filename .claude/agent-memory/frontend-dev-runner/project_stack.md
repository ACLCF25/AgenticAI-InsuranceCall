---
name: Project Stack
description: Frontend framework, tooling, and dev server details for the monolith project
type: project
---

- Framework: Next.js 16.1.6 with App Router (Turbopack enabled)
- CSS: Tailwind CSS 3.4.1
- Package manager: npm
- Dev server command: `npm run dev` inside `frontend/`
- Dev server URL: http://localhost:3000
- Backend URL: http://localhost:5000/api (set via NEXT_PUBLIC_API_URL)
- State management: TanStack React Query v5
- UI components: Radix UI primitives + shadcn/ui pattern
- HTTP client: axios (singleton in `frontend/lib/api.ts`)
- Auth: JWT via localStorage (`auth_token`), attached by axios interceptor

**Why:** Standard Next.js monolith with a Python Flask backend at port 5000.
**How to apply:** Always start from `frontend/` directory; API calls go through the `api` singleton which handles auth headers automatically.
