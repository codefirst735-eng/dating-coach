# RFH - Relationship for Humans Dating Coach

AI-powered dating coach using Gemini API with PDF knowledge base integration.

## 🚀 Quick Deploy to Vercel (Recommended)

### What You Need:
- GitHub account
- Vercel account (free - sign up at https://vercel.com)

### Deploy Steps:

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin YOUR_GITHUB_REPO_URL
   git push -u origin main
   ```

2. **Deploy on Vercel**
   - Go to https://vercel.com and sign in with GitHub
   - Click "Add New Project"
   - Import your repository
   - Add environment variable:
     - `GEMINI_API_KEY`: Your Gemini API key
   - Click "Deploy"

3. **Done!** Your app will be live at `https://your-app.vercel.app`

See `VERCEL_DEPLOYMENT.md` for detailed instructions.

## 🏃 Run Locally

### Backend
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r backend/requirements.txt
cd backend
uvicorn main:app --reload --port 8001
```

### Frontend
```bash
python3 -m http.server 8081 -d frontend
```

Visit: http://localhost:8081

## 📁 Project Structure

```
├── backend/           # FastAPI Python backend
│   ├── main.py       # Main API server
│   └── requirements.txt
├── frontend/         # Static HTML/CSS/JS
│   ├── index.html
│   ├── chat.html
│   ├── profile.html
│   └── js/
│       └── config.js # API configuration
└── vercel.json       # Vercel deployment config
```

## 🔑 Environment Variables

- `GEMINI_API_KEY`: Your Google Gemini API key

## 🎯 Features

- ✅ AI Dating Coach powered by Gemini
- ✅ PDF knowledge base upload
- ✅ User authentication & profiles
- ✅ Subscription tiers (Sleeper/Initiate/Master)
- ✅ Chat history persistence
- ✅ Screenshot analysis

## 📝 License

MIT
