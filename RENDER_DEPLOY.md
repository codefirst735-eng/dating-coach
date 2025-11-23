# Render.com Deployment (100% Free)

## Step 1: Sign Up
1. Go to https://render.com
2. Click "Get Started" → Sign up with GitHub
3. Authorize Render

## Step 2: Create Web Service
1. Click "**New +**" → "**Web Service**"
2. Connect your GitHub repository: `codefirst735-eng/dating-coach`
3. Configure:
   - **Name**: `dating-coach-backend`
   - **Region**: Choose closest to you
   - **Branch**: `main`
   - **Root Directory**: Leave empty
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r backend/requirements.txt`
   - **Start Command**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: `Free`

## Step 3: Add Environment Variable
1. Scroll down to "**Environment Variables**"
2. Click "**Add Environment Variable**"
3. Add:
   - **Key**: `GEMINI_API_KEY`
   - **Value**: `AIzaSyCldic_c7MjqgN87y_krjrOgbSUuhb9K-s`

## Step 4: Deploy
1. Click "**Create Web Service**"
2. Wait 3-5 minutes for deployment
3. You'll get a URL like: `https://dating-coach-backend.onrender.com`

## Step 5: Deploy Frontend to Cloudflare Pages
1. Go to https://dash.cloudflare.com
2. Workers & Pages → Create application → Pages
3. Connect to Git → Select your repo
4. Settings:
   - **Production branch**: `main`
   - **Build output directory**: `frontend`
   - **Build command**: (leave empty)
5. Click "Save and Deploy"

## Step 6: Update Frontend Config
After getting your Render URL, update `frontend/js/config.js`:
```javascript
const API_BASE_URL = window.location.hostname === 'localhost' 
    ? 'http://127.0.0.1:8001' 
    : 'https://dating-coach-backend.onrender.com';
```

Push the change and Cloudflare will auto-redeploy.

## ✅ Render Free Tier:
- 750 hours/month (enough for 24/7)
- Automatic HTTPS
- Auto-deploy from GitHub
- No credit card required
- Sleeps after 15 min inactivity (wakes up on first request)

## Done! 🎉
Your app will be live at your Cloudflare Pages URL!
