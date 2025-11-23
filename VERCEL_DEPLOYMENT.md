# Vercel Deployment Guide

## Quick Start (5 minutes)

### Step 1: Push to GitHub
```bash
cd "/Users/rishikohli/Desktop/coding_projects/new coding project copy"
git init
git add .
git commit -m "Initial commit"
# Create repo on GitHub, then:
git remote add origin YOUR_GITHUB_REPO_URL
git push -u origin main
```

### Step 2: Deploy on Vercel
1. Go to https://vercel.com
2. Sign in with GitHub
3. Click "Add New Project"
4. Import your GitHub repository
5. Vercel will auto-detect settings
6. Add environment variable:
   - Name: `GEMINI_API_KEY`
   - Value: `AIzaSyCldic_c7MjqgN87y_krjrOgbSUuhb9K-s`
7. Click "Deploy"

### Step 3: Update Frontend Config
After deployment, Vercel gives you a URL like: `https://your-app.vercel.app`

Update `frontend/js/config.js`:
```javascript
const API_BASE_URL = window.location.origin + '/api';
```

Then push changes:
```bash
git add .
git commit -m "Update API config"
git push
```

Vercel will automatically redeploy!

## That's It! 🎉

Your app will be live at: `https://your-app.vercel.app`

- Frontend: `https://your-app.vercel.app`
- Backend API: `https://your-app.vercel.app/api/*`

## Custom Domain (Optional)
In Vercel dashboard → Settings → Domains → Add your domain

## Notes
- Every git push triggers auto-deployment
- Free tier is very generous
- Vercel handles HTTPS automatically
- Database (SQLite) will reset on each deployment - consider upgrading to PostgreSQL for persistence
