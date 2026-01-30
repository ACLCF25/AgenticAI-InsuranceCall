# Autonomous AI Insurance Credentialing System

A complete full-stack application combining Next.js 14 frontend with Python/LangChain/LangSmith backend for autonomous insurance credentialing phone calls.

## Architecture

```
+----------------------------------------------------------+
|                    Next.js Frontend                       |
|  [Dashboard] [Calls] [Follow-ups] [Analytics]            |
|  React Query - Tailwind CSS - shadcn/ui - TypeScript     |
+----------------------------------------------------------+
                           |
                    REST API / WebSocket
                           |
+----------------------------------------------------------+
|                  Python Flask Backend                     |
|  +----------------------------------------------------+  |
|  |            LangGraph State Machine                 |  |
|  |  Init -> Classify -> Navigate -> Converse -> Extract  |
|  +----------------------------------------------------+  |
|  LangChain - LangSmith - OpenAI GPT-4 - PostgreSQL       |
+----------------------------------------------------------+
                           |
         +-----------------+-----------------+
         |                 |                 |
    [Twilio]          [Deepgram]        [Supabase]
     Phone               STT            PostgreSQL
```

## Project Structure

```
monolith/
├── frontend/                    # Next.js 14 Application
│   ├── app/
│   │   ├── layout.tsx          # Root layout
│   │   ├── page.tsx            # Dashboard page
│   │   ├── providers.tsx       # React Query provider
│   │   └── globals.css         # Global styles
│   ├── components/
│   │   ├── dashboard/          # Dashboard components
│   │   └── ui/                 # shadcn/ui components
│   ├── lib/
│   │   ├── api.ts              # API client
│   │   └── utils.ts            # Utility functions
│   └── types/
│       └── index.ts            # TypeScript types
│
├── backend/                     # Python Flask Backend
│   ├── credentialing_agent.py  # LangGraph agent
│   ├── api_server.py           # Flask API
│   ├── cli.py                  # CLI tool
│   ├── requirements.txt
│   └── .env.example
│
├── database/
│   └── supabase_schema.sql     # Database schema
│
└── docker-compose.yml          # Full stack deployment
```

## Quick Start

### Prerequisites

- **Node.js 18+** and **npm**
- **Python 3.9+**
- **PostgreSQL** (via Supabase)
- API Keys: LangSmith, OpenAI (GPT-4), Twilio, Deepgram, ElevenLabs

### Installation

**1. Clone and Setup**

```bash
# Clone repository
git clone https://github.com/ACLCF25/AgenticAI-InsuranceCall.git
cd AgenticAI-InsuranceCall

# Setup backend
cd backend
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys

# Setup frontend
cd ../frontend
npm install
```

**2. Database Setup**

1. Go to [Supabase](https://supabase.com)
2. Create new project
3. Open SQL Editor
4. Copy/paste contents of `database/supabase_schema.sql`
5. Execute

**3. Configure Environment**

Backend (`.env`):
```
LANGSMITH_API_KEY=lsv2_...
OPENAI_API_KEY=sk-...
TWILIO_ACCOUNT_SID=AC...
TWILIO_AUTH_TOKEN=...
DEEPGRAM_API_KEY=...
ELEVENLABS_API_KEY=...
SUPABASE_PASSWORD=...
```

Frontend (`.env.local`):
```
NEXT_PUBLIC_API_URL=http://localhost:5000/api
NEXT_PUBLIC_WS_URL=ws://localhost:5000
```

**4. Run Development Servers**

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

## Features

### Frontend
- **Real-time Dashboard** - Live call status, metrics, active calls monitoring
- **Call Management** - Start calls, view history, track progress, transcripts
- **Follow-ups** - Automated scheduling, manual execution, status tracking
- **Analytics** - Success rates, duration stats, provider performance, costs
- **IVR Knowledge Base** - View/add IVR patterns, track success rates

### Backend
- **Autonomous AI Agent** - LangGraph state machine with LangSmith tracing
- **Call Capabilities** - IVR navigation, hold detection, natural conversation
- **Learning System** - Updates IVR success rates, improves over time

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/start-call` | Start new credentialing call |
| GET | `/api/call-status/:id` | Get call status |
| GET | `/api/call-transcript/:id` | Get transcript |
| GET | `/api/metrics` | System metrics |
| GET | `/api/scheduled-followups` | Pending follow-ups |
| POST | `/api/ivr-knowledge` | Add IVR knowledge |
| GET | `/api/health` | Health check |

## Docker Deployment

```bash
docker-compose up -d
```

This starts:
- Next.js frontend (port 3000)
- Python backend (port 5000)

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | Next.js 14, React Query, Tailwind CSS, shadcn/ui, TypeScript |
| Backend | Python, Flask, LangChain, LangGraph, LangSmith |
| Database | PostgreSQL (Supabase) |
| Voice | Twilio (calls), Deepgram (STT), ElevenLabs (TTS) |
| AI | OpenAI GPT-4 |

## Cost Estimation

Per call (15 min average):
- OpenAI GPT-4: ~$0.40
- Deepgram STT: ~$0.06
- ElevenLabs TTS: ~$0.15
- Twilio: ~$0.20
- **Total: ~$0.81**

## Monitoring

All backend operations are traced in LangSmith:
1. Go to https://smith.langchain.com
2. Select project: `insurance-credentialing-agent`
3. View traces, costs, latency, decisions

## License

MIT License

---

**Built with** Next.js 14 | Python | LangChain | LangSmith | Supabase
