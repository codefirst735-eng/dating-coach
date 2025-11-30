# ⚡ Quick Start - RFH

## On Your New Machine

### 1. Clone & Setup (First Time Only)
```bash
cd ~/Desktop/coding_projects
git clone https://github.com/codefirst735-eng/dating-coach.git
cd dating-coach
git checkout development
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

### 2. Create .env File
```bash
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

### 3. Run Application (Every Time)

**Terminal 1 - Backend:**
```bash
cd ~/Desktop/coding_projects/dating-coach
source venv/bin/activate
venv/bin/python -m uvicorn backend.main:app --reload --port 8001
```

**Terminal 2 - Frontend:**
```bash
cd ~/Desktop/coding_projects/dating-coach
python3 -m http.server 8082 -d frontend
```

**Open Browser:**
```
http://127.0.0.1:8082
```

---

## Common Commands

### Git
```bash
# Pull latest code
git pull origin development

# Commit & push changes
git add .
git commit -m "your message"
git push origin development
```

### Stop Servers
```bash
# Kill backend (port 8001)
lsof -ti:8001 | xargs kill -9

# Kill frontend (port 8082)
lsof -ti:8082 | xargs kill -9
```

### Reset Database
```bash
rm users.db
# Restart backend to recreate
```

---

## URLs

- **Frontend**: http://127.0.0.1:8082
- **Backend API**: http://127.0.0.1:8001
- **API Docs**: http://127.0.0.1:8001/docs
- **Admin Panel**: http://127.0.0.1:8082/admin.html

---

## Files to Configure

1. **.env** - Add your Gemini API key
2. **js/config.js** - API URL configuration (already set for local)

---

## Troubleshooting

- ❌ Port in use? → Run kill commands above
- ❌ Module errors? → Activate venv and reinstall: `pip install -r backend/requirements.txt`
- ❌ API errors? → Check `.env` has valid `GEMINI_API_KEY`
- ❌ Database errors? → Delete `users.db` and restart backend

---

For detailed setup, see [SETUP.md](./SETUP.md)
