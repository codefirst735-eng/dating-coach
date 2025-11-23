# Railway Deployment Guide

## Quick Deploy (5 minutes)

### Step 1: Sign up for Railway
1. Go to https://railway.app
2. Sign in with GitHub

### Step 2: Deploy Backend
1. Click "New Project"
2. Select "Deploy from GitHub repo"
3. Choose your `dating-coach` repository
4. Railway will auto-detect Python
5. Add environment variable:
   - Key: `GEMINI_API_KEY`
   - Value: `AIzaSyCldic_c7MjqgN87y_krjrOgbSUuhb9K-s`
6. Click "Deploy"

### Step 3: Get Your Backend URL
After deployment, Railway gives you a URL like:
`https://dating-coach-production.up.railway.app`

### Step 4: Deploy Frontend to Cloudflare Pages
1. Go to https://dash.cloudflare.com
2. Pages → Create a project
3. Connect GitHub → Select your repo
4. Settings:
   - Build output directory: `frontend`
   - Build command: (leave empty)
5. Deploy

### Step 5: Update Frontend Config
Edit `frontend/js/config.js` and update with your Railway URL:
```javascript
const API_BASE_URL = window.location.hostname === 'localhost' 
    ? 'http://127.0.0.1:8001' 
    : 'https://your-app.up.railway.app';
```

Push changes and Cloudflare will auto-redeploy.

## Done! 🎉

- Backend: Railway (Python works perfectly)
- Frontend: Cloudflare Pages (fast CDN)

Both are free tier and work great together!
