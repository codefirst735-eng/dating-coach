# Git Workflow Guide

## 📌 Current Setup

### Branches:
- **`main`** - Production branch (auto-deploys to Vercel + Render)
- **`development`** - Development branch (local testing only)

### Current Stable Commit:
**Hash**: `6ab5369663bcdfeda71be539db89f73a196bc9d3`
**Message**: Fix guest-chat to use Gemini API
**Date**: 2025-11-24

---

## 🔄 Workflow

### Working on New Features (Development Branch)

1. **Make sure you're on development branch:**
   ```bash
   git checkout development
   ```

2. **Make your changes and commit:**
   ```bash
   git add .
   git commit -m "Your commit message"
   ```

3. **Push to development (won't trigger deployment):**
   ```bash
   git push origin development
   ```

4. **Test locally:**
   - Backend: `venv/bin/python -m uvicorn backend.main:app --reload --port 8001`
   - Frontend: `python3 -m http.server 8081 -d frontend`

---

### Deploying to Production

When you're ready to deploy your changes:

1. **Switch to main branch:**
   ```bash
   git checkout main
   ```

2. **Merge development into main:**
   ```bash
   git merge development
   ```

3. **Push to main (triggers auto-deployment):**
   ```bash
   git push origin main
   ```

4. **Vercel and Render will auto-deploy** (wait 2-3 minutes)

---

### Switching Between Branches

**Go to development (for new work):**
```bash
git checkout development
```

**Go to main (to deploy or check production):**
```bash
git checkout main
```

**Check which branch you're on:**
```bash
git branch
```
(The branch with `*` is your current branch)

---

## 🔙 Emergency Rollback

If something breaks in production, reset to the stable commit:

```bash
git checkout main
git reset --hard 6ab5369663bcdfeda71be539db89f73a196bc9d3
git push --force origin main
```

⚠️ **Warning**: This will discard all commits after the stable point!

---

## 📋 Quick Reference

| Action | Command |
|--------|---------|
| Check current branch | `git branch` |
| Switch to development | `git checkout development` |
| Switch to main | `git checkout main` |
| Commit changes | `git add . && git commit -m "message"` |
| Push development | `git push origin development` |
| Deploy to production | `git checkout main && git merge development && git push origin main` |
| View commit history | `git log --oneline` |

---

## 🎯 Current Status

- ✅ You are now on: **`development`** branch
- ✅ All future changes will be on development
- ✅ Main branch is stable and deployed
- ✅ Stable commit hash saved: `6ab5369`

---

## 💡 Tips

1. **Always work on development branch** for new features
2. **Test thoroughly locally** before merging to main
3. **Merge to main only when ready to deploy**
4. **Keep commits small and descriptive**
5. **If unsure, check current branch**: `git branch`
