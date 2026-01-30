# Autonomous AI Insurance Credentialing System
## Monolith: Next.js Frontend + Python Backend

A complete full-stack application combining Next.js 14 frontend with Python/LangChain/LangSmith backend for autonomous insurance credentialing phone calls.

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      Next.js Frontend                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │Dashboard │  │  Calls   │  │Follow-ups│  │Analytics │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
│                                                               │
│  React Query • Tailwind CSS • shadcn/ui • TypeScript        │
└──────────────────────────────────────────────────────────────┘
                            │
                      REST API / WebSocket
                            │
┌──────────────────────────────────────────────────────────────┐
│                   Python Flask Backend                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              LangGraph State Machine                  │   │
│  │  Init → Classify → Navigate → Converse → Extract     │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                               │
│  LangChain • LangSmith • OpenAI GPT-4 • PostgreSQL          │
└──────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
  ┌──────────┐      ┌──────────┐       ┌──────────┐
  │ Twilio   │      │Deepgram  │       │PostgreSQL│
  │  Phone   │      │   STT    │       │ Supabase │
  └──────────┘      └──────────┘       └──────────┘
```

## 📁 Project Structure

```
credentialing-monolith/
├── frontend/                    # Next.js 14 Application
│   ├── app/
│   │   ├── layout.tsx          # Root layout
│   │   ├── page.tsx            # Dashboard page
│   │   ├── providers.tsx       # React Query provider
│   │   └── globals.css         # Global styles
│   ├── components/
│   │   ├── dashboard/
│   │   │   ├── header.tsx
│   │   │   ├── stats-cards.tsx
│   │   │   ├── active-calls-table.tsx
│   │   │   ├── recent-calls-table.tsx
│   │   │   ├── start-call-dialog.tsx
│   │   │   ├── followups-panel.tsx
│   │   │   └── metrics-chart.tsx
│   │   └── ui/                 # shadcn/ui components
│   ├── lib/
│   │   ├── api.ts              # API client
│   │   └── utils.ts            # Utility functions
│   ├── types/
│   │   └── index.ts            # TypeScript types
│   ├── package.json
│   ├── tsconfig.json
│   ├── tailwind.config.js
│   └── next.config.js
│
├── backend/                     # Python Flask Backend
│   ├── credentialing_agent.py  # LangGraph agent
│   ├── api_server.py           # Flask API
│   ├── cli.py                  # CLI tool
│   ├── requirements.txt
│   └── .env
│
├── database/
│   └── supabase_schema.sql     # Database schema
│
├── docker-compose.yml          # Full stack deployment
└── README.md                   # This file
```

## 🚀 Quick Start (5 minutes)

### Prerequisites

- **Node.js 18+** and **npm**
- **Python 3.9+**
- **PostgreSQL** (via Supabase)
- API Keys:
  - LangSmith
  - OpenAI (GPT-4)
  - Twilio
  - Deepgram
  - ElevenLabs

### Installation

#### 1. Clone and Setup

```bash
# Clone repository
git clone <repo-url>
cd credentialing-monolith

# Setup backend
cd backend
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.template .env
# Edit .env with your API keys

# Setup frontend
cd ../frontend
npm install
```

#### 2. Database Setup

```bash
# Go to Supabase (https://supabase.com)
# Create new project
# Open SQL Editor
# Copy/paste contents of database/supabase_schema.sql
# Execute
```

#### 3. Configure Environment

**Backend (.env):**
```bash
LANGSMITH_API_KEY=lsv2_...
OPENAI_API_KEY=sk-...
TWILIO_ACCOUNT_SID=AC...
TWILIO_AUTH_TOKEN=...
DEEPGRAM_API_KEY=...
ELEVENLABS_API_KEY=...
SUPABASE_PASSWORD=...
```

**Frontend (.env.local):**
```bash
NEXT_PUBLIC_API_URL=http://localhost:5000/api
NEXT_PUBLIC_WS_URL=ws://localhost:5000
```

#### 4. Run Development Servers

```bash
# Terminal 1 - Backend
cd backend
source venv/bin/activate
python api_server.py

# Terminal 2 - Frontend
cd frontend
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

## 🎯 Features

### Frontend Features

✅ **Real-time Dashboard**
- Live call status updates
- System metrics and analytics
- Active calls monitoring

✅ **Call Management**
- Start new credentialing calls
- View call history
- Track call progress in real-time
- View full conversation transcripts

✅ **Follow-ups Management**
- Automated follow-up scheduling
- Manual follow-up execution
- Follow-up status tracking

✅ **Analytics**
- Success rate trends
- Call duration statistics
- Insurance provider performance
- Cost tracking

✅ **IVR Knowledge Base**
- View learned IVR patterns
- Add manual IVR knowledge
- Track success rates per pattern

### Backend Features

✅ **Autonomous AI Agent**
- LangGraph state machine
- LangSmith tracing for all decisions
- PostgreSQL checkpoint persistence

✅ **Call Capabilities**
- IVR navigation (DTMF + speech)
- Hold detection and patience
- Natural human conversation
- Structured data extraction

✅ **Learning System**
- Updates IVR success rates
- Learns from each call
- Improves over time

## 🖥️ Frontend Pages

### Dashboard (`/`)
Main dashboard with:
- Quick stats cards
- Active calls table
- Recent calls list
- Pending follow-ups
- Start call button

### Call Details (`/calls/[id]`)
Detailed call view with:
- Call status and timeline
- Full conversation transcript
- Extracted information
- LangSmith trace link

### Analytics (`/analytics`)
- Success rate charts
- Call duration trends
- Provider performance
- Cost analysis

### IVR Knowledge (`/ivr-knowledge`)
- Browse IVR patterns by insurance
- Add new patterns manually
- View success rates
- Export knowledge base

## 🔧 Configuration

### Frontend API Client

Located in `frontend/lib/api.ts`:

```typescript
// Automatically connects to backend
const api = new APIClient()

// Start a call
await api.startCall(requestData)

// Get status
await api.getCallStatus(callId)

// Get metrics
await api.getMetrics()
```

### Backend Endpoints

```
POST   /api/start-call          # Start new call
GET    /api/call-status/:id     # Get call status
GET    /api/call-transcript/:id # Get transcript
GET    /api/metrics             # System metrics
GET    /api/scheduled-followups # Pending follow-ups
POST   /api/ivr-knowledge       # Add IVR knowledge
GET    /api/health              # Health check
```

## 🎨 UI Components

Built with **shadcn/ui** and **Tailwind CSS**:

- `Button` - Primary actions
- `Card` - Content containers
- `Dialog` - Modals (Start Call)
- `Table` - Data display
- `Tabs` - Navigation
- `Form` - Input handling
- `Toast` - Notifications
- `Select` - Dropdowns
- `Tooltip` - Hints

All components fully typed with TypeScript.

## 📊 State Management

### React Query

Used for server state:

```typescript
// Fetch with auto-refresh
const { data, isLoading } = useQuery({
  queryKey: ['metrics'],
  queryFn: () => api.getMetrics(),
  refetchInterval: 30000, // 30s
})

// Mutations
const mutation = useMutation({
  mutationFn: api.startCall,
  onSuccess: () => {
    queryClient.invalidateQueries(['calls'])
  }
})
```

### WebSocket (Optional)

Real-time updates:

```typescript
import io from 'socket.io-client'

const socket = io(process.env.NEXT_PUBLIC_WS_URL)

socket.on('call_update', (data) => {
  // Update UI in real-time
})
```

## 🐳 Docker Deployment

### Single Command Deployment

```bash
docker-compose up -d
```

This starts:
- Next.js frontend (port 3000)
- Python backend (port 5000)
- PostgreSQL (managed by Supabase)

### docker-compose.yml

```yaml
version: '3.8'

services:
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://backend:5000/api
    depends_on:
      - backend

  backend:
    build: ./backend
    ports:
      - "5000:5000"
    environment:
      - LANGSMITH_API_KEY=${LANGSMITH_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      # ... other env vars
```

## 🧪 Development

### Frontend Development

```bash
cd frontend

# Run dev server
npm run dev

# Type checking
npm run type-check

# Lint
npm run lint

# Build for production
npm run build
npm start
```

### Backend Development

```bash
cd backend

# Run server
python api_server.py

# Run tests
pytest tests/

# CLI commands
python cli.py metrics
python cli.py list
```

## 📈 Monitoring

### LangSmith Dashboard

Every backend operation is traced:

1. Go to https://smith.langchain.com
2. Select project: `insurance-credentialing-agent`
3. View traces in real-time
4. See costs, latency, decisions

### Frontend Monitoring

Built-in error boundary and toast notifications for all operations.

## 🔐 Security

- Environment variables for secrets
- CORS configuration
- API request validation
- XSS protection (React)
- SQL injection prevention (parameterized queries)
- Rate limiting on backend

## 💰 Cost Estimation

Per call (15 min average):

- **OpenAI GPT-4**: ~$0.40
- **Deepgram STT**: ~$0.06
- **ElevenLabs TTS**: ~$0.15
- **Twilio**: ~$0.20
- **Total**: ~$0.81

Hosting costs:
- **Vercel (Frontend)**: Free tier
- **Railway (Backend)**: ~$5/mo
- **Supabase**: Free tier

## 🚀 Production Deployment

### Option 1: Vercel + Railway

**Frontend (Vercel):**
```bash
cd frontend
vercel --prod
```

**Backend (Railway):**
```bash
cd backend
railway up
```

### Option 2: Single VPS

```bash
# Clone repo on VPS
git clone <repo>

# Run with PM2
pm2 start ecosystem.config.js
```

### Option 3: Kubernetes

```bash
kubectl apply -f k8s/
```

## 📚 Documentation

- **Frontend API Docs**: `frontend/lib/api.ts`
- **Backend API Docs**: `backend/api_server.py`
- **Type Definitions**: `frontend/types/index.ts`
- **Database Schema**: `database/supabase_schema.sql`

## 🛠️ Troubleshooting

### Frontend Issues

```bash
# Clear Next.js cache
rm -rf .next

# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install

# Check API connection
curl http://localhost:5000/api/health
```

### Backend Issues

```bash
# Verify Python environment
python --version
pip list

# Test database connection
python -c "from credentialing_agent import DatabaseManager; db = DatabaseManager(); db.close()"

# Check logs
tail -f logs/app.log
```

### Common Errors

**CORS Error:**
- Update `CORS_ORIGINS` in backend
- Add frontend URL to allowed origins

**API Connection Failed:**
- Verify `NEXT_PUBLIC_API_URL` in frontend
- Check backend is running on port 5000
- Verify firewall rules

**Database Connection:**
- Check Supabase credentials
- Verify network access to Supabase

## 🎓 Learning Resources

- **Next.js**: https://nextjs.org/docs
- **React Query**: https://tanstack.com/query
- **LangChain**: https://python.langchain.com
- **LangGraph**: https://langchain-ai.github.io/langgraph/
- **Tailwind CSS**: https://tailwindcss.com
- **shadcn/ui**: https://ui.shadcn.com

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Add tests
5. Submit pull request

## 📝 License

MIT License

## 🆘 Support

- **GitHub Issues**: Report bugs
- **Discussions**: Ask questions
- **Email**: support@yourcompany.com

---

**Built with** ⚡ **Next.js 14** | 🐍 **Python** | 🦜🔗 **LangChain** | 📊 **LangSmith**

*Complete autonomous credentialing solution - from UI to AI agent*
#   A g e n t i c A I - I n s u r a n c e C a l l  
 