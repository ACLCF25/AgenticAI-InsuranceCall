# Quick Start Guide
## Autonomous Credentialing System - Monolith

Get up and running in 5 minutes!

## 📋 Prerequisites Checklist

Before starting, ensure you have:

- [ ] **Node.js 18+** installed (`node --version`)
- [ ] **Python 3.9+** installed (`python3 --version`)
- [ ] **Supabase account** created (https://supabase.com)
- [ ] **API Keys** ready:
  - [ ] LangSmith API key (https://smith.langchain.com)
  - [ ] OpenAI API key with GPT-4 access
  - [ ] Twilio account with phone number
  - [ ] Deepgram API key
  - [ ] ElevenLabs API key

## 🚀 Installation (3 commands)

```bash
# 1. Run the setup script
chmod +x setup.sh
./setup.sh

# 2. Setup database (follow prompts in setup script)
# - Go to Supabase
# - Create project
# - Run database/supabase_schema.sql in SQL Editor

# 3. Update API keys
# Edit backend/.env with your API keys
nano backend/.env
```

## ▶️ Running the Application

### Development Mode

**Terminal 1 - Backend:**
```bash
cd backend
source venv/bin/activate
python api_server.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

**Open:** http://localhost:3000

### Production Mode (Docker)

```bash
# Start everything
docker-compose up -d

# View logs
docker-compose logs -f

# Stop everything
docker-compose down
```

## 🎯 First Steps

1. **Access Dashboard**: http://localhost:3000
2. **Click "Start New Call"**
3. **Fill in the form:**
   - Insurance: Blue Cross Blue Shield
   - Provider: Dr. Jane Smith
   - NPI: 1234567890
   - Tax ID: 12-3456789
   - Address: 123 Main St, City, ST 12345
   - Phone: +18001234567
   - Questions: What is the credentialing status?
4. **Click "Start Call"**
5. **Monitor progress** in the dashboard

## 📊 What You'll See

### Dashboard Features

- **Stats Cards**: Total calls, success rate, etc.
- **Active Calls Table**: Real-time call status
- **Recent Calls**: Historical data
- **Follow-ups Panel**: Scheduled callbacks

### Call Flow

1. **Initiating** → Call is being placed
2. **IVR Navigation** → Navigating menu system
3. **On Hold** → Waiting for representative
4. **Speaking with Human** → Active conversation
5. **Extracting Info** → Pulling structured data
6. **Completing** → Finalizing results

## 🔍 Monitoring with LangSmith

1. Go to https://smith.langchain.com
2. Select your project: `insurance-credentialing-agent`
3. See every decision the AI makes
4. View costs, latency, and success rates

## 🛠️ Configuration

### Backend (.env)

```bash
# Essential - must configure
LANGSMITH_API_KEY=lsv2_...
OPENAI_API_KEY=sk-...
TWILIO_ACCOUNT_SID=AC...
TWILIO_AUTH_TOKEN=...
DEEPGRAM_API_KEY=...
ELEVENLABS_API_KEY=...
SUPABASE_PASSWORD=...

# Optional - defaults work
MAX_HOLD_TIME_MINUTES=30
MAX_RETRY_ATTEMPTS=3
```

### Frontend (.env.local)

```bash
# Automatically created by setup.sh
NEXT_PUBLIC_API_URL=http://localhost:5000/api
NEXT_PUBLIC_WS_URL=ws://localhost:5000
```

## 📱 Twilio Webhook Setup

1. Go to Twilio Console
2. Select your phone number
3. Configure Voice:
   - **A Call Comes In**: `https://your-domain.com/webhook/voice`
   - **Status Callback**: `https://your-domain.com/webhook/status`

For development, use ngrok:
```bash
ngrok http 5000
# Use the ngrok HTTPS URL in Twilio
```

## ✅ Testing the System

### Test Backend
```bash
cd backend
source venv/bin/activate
python cli.py metrics
```

### Test Frontend
```bash
cd frontend
npm run build
npm start
```

### Test Full Integration
```bash
# Start call via CLI
cd backend
python cli.py start --file ../test-data/sample-call.json

# Or use the web UI
# Open http://localhost:3000
# Click "Start New Call"
```

## 🐛 Troubleshooting

### Backend won't start

```bash
# Check Python version
python3 --version  # Should be 3.9+

# Check dependencies
cd backend
source venv/bin/activate
pip list

# Test database
python -c "from credentialing_agent import DatabaseManager; db = DatabaseManager(); print('OK'); db.close()"
```

### Frontend won't start

```bash
# Clear cache
cd frontend
rm -rf .next node_modules
npm install
npm run dev

# Check Node version
node --version  # Should be 18+
```

### Can't connect to API

```bash
# Check backend is running
curl http://localhost:5000/api/health

# Check frontend .env.local
cat .env.local
# Should point to http://localhost:5000/api
```

### Database connection error

1. Verify Supabase credentials in `backend/.env`
2. Check Supabase is accessible: https://supabase.com
3. Ensure SQL schema was executed
4. Test connection:
   ```bash
   cd backend
   source venv/bin/activate
   python -c "from credentialing_agent import DatabaseManager; DatabaseManager().close()"
   ```

## 📚 Next Steps

Once running:

1. **Explore Dashboard**: See all features
2. **Make Test Calls**: Start with known insurance providers
3. **View LangSmith**: Watch AI decisions
4. **Add IVR Knowledge**: Help the system learn
5. **Schedule Follow-ups**: Test automation
6. **Check Analytics**: Review performance

## 🎓 Learn More

- **Frontend Code**: `frontend/components/dashboard/`
- **Backend Code**: `backend/credentialing_agent.py`
- **API Client**: `frontend/lib/api.ts`
- **Type Definitions**: `frontend/types/index.ts`
- **Database Schema**: `database/supabase_schema.sql`

## 💡 Pro Tips

1. **Use LangSmith religiously** - it shows exactly what the AI is thinking
2. **Start with simple calls** - test with offices that answer quickly
3. **Monitor costs** - check OpenAI dashboard regularly
4. **Build IVR knowledge** - add patterns as you discover them
5. **Use the CLI** - faster for bulk operations

## 🆘 Getting Help

1. **Check logs**:
   - Backend: `backend/logs/app.log`
   - Frontend: Browser console (F12)

2. **Review traces**:
   - LangSmith dashboard
   - Twilio call logs

3. **Test components individually**:
   ```bash
   # Test backend only
   cd backend
   python cli.py metrics
   
   # Test frontend only
   cd frontend
   npm run dev
   ```

## ✨ Success Indicators

You know it's working when:

- ✅ Dashboard shows stats
- ✅ Can start a new call
- ✅ Backend logs show activity
- ✅ LangSmith shows traces
- ✅ Database has records

## 🚀 Production Deployment

Ready for production? See:

- **README.md** - Full deployment guide
- **docker-compose.yml** - Container setup
- **Vercel + Railway** - Easiest deployment
- **AWS/GCP/Azure** - Enterprise options

---

**Questions?** Open an issue or check the full README.md
