# 🔧 LOCAL DEVELOPMENT MODE

## Current Configuration

✅ **Branch:** `development`  
✅ **Backend:** `http://127.0.0.1:8001` (Local)  
✅ **Frontend:** `http://127.0.0.1:8082` (Local)  
✅ **Auto-deployment:** DISABLED for development branch

---

## Running Locally

### 1. Start Backend Server (Terminal 1)
```bash
cd ~/Desktop/coding_projects/new\ coding\ project\ copy
source venv/bin/activate
venv/bin/python -m uvicorn backend.main:app --reload --port 8001
```

### 2. Start Frontend Server (Terminal 2)
```bash
cd ~/Desktop/coding_projects/new\ coding\ project\ copy
python3 -m http.server 8082 -d frontend
```

### 3. Access Application
- **Frontend:** http://127.0.0.1:8082
- **Backend API:** http://127.0.0.1:8001
- **API Docs:** http://127.0.0.1:8001/docs

---

## Development Workflow

### Making Changes
1. Work on the `development` branch
2. Test changes locally at http://127.0.0.1:8082
3. Commit changes: `git add . && git commit -m "your message"`
4. Push to development: `git push origin development`

**Note:** Pushing to `development` will NOT trigger deployments on Vercel or Render.

### Deploying to Production

**Before merging to main:**
1. Update `frontend/js/config.js`:
   ```javascript
   // Comment out the development config:
   // const API_BASE_URL = 'http://127.0.0.1:8001';
   
   // Uncomment the production config:
   const API_BASE_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
       ? 'http://127.0.0.1:8001'
       : 'https://dating-coach-ytos.onrender.com';
   ```

2. Test locally one more time
3. Commit the production config change
4. Merge to main:
   ```bash
   git checkout main
   git merge development
   git push origin main
   ```

5. **Vercel and Render will auto-deploy from main branch only**

6. After deployment, switch back to development:
   ```bash
   git checkout development
   ```

---

## Deployment Protection

### Vercel Configuration (`vercel.json`)
- ✅ Deploys from `main` branch
- ❌ Does NOT deploy from `development` branch

### Render Configuration (`render.yaml`)
- ✅ Deploys from `main` branch
- ❌ Does NOT deploy from `development` branch

---

## Important Files

- `frontend/js/config.js` - API URL configuration
- `vercel.json` - Vercel deployment settings
- `render.yaml` - Render deployment settings
- `.deployment-lock` - Reminder about local dev mode

---

## Common Commands

```bash
# Switch to development
git checkout development

# See current branch
git branch

# Pull latest changes
git pull origin development

# Push changes (won't deploy)
git push origin development

# Check if servers are running
lsof -ti:8001  # Backend
lsof -ti:8082  # Frontend

# Kill servers if needed
lsof -ti:8001 | xargs kill -9
lsof -ti:8082 | xargs kill -9
```

---

## Troubleshooting

### Frontend can't connect to backend
- Check that both servers are running
- Verify `frontend/js/config.js` points to `http://127.0.0.1:8001`

### Changes not showing up
- Hard refresh browser: `Cmd + Shift + R` (Mac) or `Ctrl + Shift + R` (Windows)
- Clear browser cache

### Database issues
- Delete `users.db` and restart backend to reset database

---

**Last Updated:** 2025-11-30  
**Mode:** Local Development Only
