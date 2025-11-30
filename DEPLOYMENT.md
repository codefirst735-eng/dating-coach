# Deployment Guide

## Overview
This application consists of two parts:
1. **Frontend** (HTML/CSS/JS) - Deploy to Cloudflare Pages
2. **Backend** (FastAPI/Python) - Deploy to Render.com or Railway.app

## Part 1: Deploy Backend (Render.com - Free Tier)

### Prerequisites
- GitHub account
- Render.com account (sign up at https://render.com)

### Steps

1. **Push your code to GitHub**
   ```bash
   cd "/Users/rishikohli/Desktop/coding_projects/new coding project copy"
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin YOUR_GITHUB_REPO_URL
   git push -u origin main
   ```

2. **Deploy on Render**
   - Go to https://dashboard.render.com
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Configure:
     - **Name**: `rfh-backend` (or your choice)
     - **Root Directory**: `backend`
     - **Environment**: `Python 3`
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
     - **Instance Type**: Free

3. **Add Environment Variables**
   In Render dashboard, add:
   - `GEMINI_API_KEY`: Your Gemini API key

4. **Note your backend URL**
   After deployment, you'll get a URL like: `https://rfh-backend.onrender.com`

## Part 2: Deploy Frontend (Cloudflare Pages)

### Prerequisites
- Cloudflare account (sign up at https://cloudflare.com)
- GitHub account

### Steps

1. **Update API URL in frontend**
   - Edit `frontend/js/config.js`
   - Replace `your-backend-url.onrender.com` with your actual Render URL

2. **Push changes to GitHub**
   ```bash
   git add .
   git commit -m "Update backend URL"
   git push
   ```

   - Click "Save and Deploy"

### Option B: Deploy Frontend (Vercel)

1. **Update API URL in frontend**
   - Ensure `frontend/js/config.js` points to your Render URL

2. **Deploy on Vercel**
   - Go to https://vercel.com
   - Click "Add New..." → "Project"
   - Import your GitHub repository
   - Configure:
     - **Framework Preset**: Other
     - **Root Directory**: `frontend` (Click Edit to change)
   - Click "Deploy"


4. **Configure CORS on Backend**
   After deployment, note your Cloudflare Pages URL (e.g., `https://rfh-app.pages.dev`)
   - Update `backend/main.py` CORS settings to include your frontend URL
   - Push changes and Render will auto-redeploy

## Part 3: Database Persistence

The current SQLite database won't persist on Render's free tier. Options:

1. **Use Render's PostgreSQL** (Recommended for production)
   - Add PostgreSQL database in Render
   - Update `backend/main.py` to use PostgreSQL instead of SQLite
   - Install `psycopg2-binary` in requirements.txt

2. **Use Render Disk** (Simple but limited)
   - Add a persistent disk in Render dashboard
   - Mount it to store `users.db`

## Environment Variables Needed

### Backend (Render)
- `GEMINI_API_KEY`: Your Gemini API key

### Frontend (Cloudflare Pages)
- None (all config in code)

## Post-Deployment Checklist

- [ ] Backend is accessible at your Render URL
- [ ] Frontend is accessible at your Cloudflare Pages URL
- [ ] CORS is configured correctly
- [ ] API calls from frontend to backend work
- [ ] User registration/login works
- [ ] Chat functionality works
- [ ] PDF upload works (if using persistent storage)

## Troubleshooting

### CORS Errors
- Ensure your Cloudflare Pages URL is in the CORS allow_origins list in `backend/main.py`

### API Connection Errors
- Verify `frontend/js/config.js` has the correct backend URL
- Check browser console for errors

### Database Issues
- Render free tier restarts periodically, losing SQLite data
- Consider upgrading to PostgreSQL for persistence
