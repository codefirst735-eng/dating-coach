# Troubleshooting PDF Upload Issues

## Common Issues and Solutions

### 1. "Uploading files..." but nothing happens

**Possible Causes:**
- Backend server not running
- OpenAI API key not configured
- Network/CORS issues
- File upload format issue

**Debugging Steps:**

1. **Check Backend Server:**
   ```bash
   # Make sure backend is running
   cd backend
   uvicorn main:app --reload
   ```

2. **Check OpenAI API Key:**
   - Create `.env` file in project root (same level as backend folder)
   - Add: `OPENAI_API_KEY=your_actual_api_key_here`
   - Restart the backend server

3. **Check Browser Console:**
   - Open browser DevTools (F12)
   - Go to Console tab
   - Try uploading a file
   - Look for error messages

4. **Check Backend Logs:**
   - Look at the terminal where backend is running
   - You should see log messages like:
     - "Received upload request for file: ..."
     - "Reading file content..."
     - "Uploading to OpenAI..."

5. **Test Backend Health:**
   - Visit: `http://127.0.0.1:8000/admin/health`
   - Should return JSON with `openai_configured: true`

### 2. "OpenAI API not configured" Error

**Solution:**
- Create `.env` file in project root
- Add your OpenAI API key:
  ```
  OPENAI_API_KEY=sk-...
  ```
- Get your key from: https://platform.openai.com/api-keys
- Restart backend server

### 3. "Invalid API key" Error

**Solution:**
- Verify your API key is correct
- Make sure there are no extra spaces in `.env` file
- Check that `.env` file is in the project root (not in backend folder)
- Restart backend server after changing `.env`

### 4. Network/CORS Errors

**Solution:**
- Make sure frontend is running on `http://localhost:8080` or `http://127.0.0.1:8080`
- Backend CORS is configured for these origins
- Check browser console for CORS errors

### 5. File Upload Format Issues

**If you see errors about file format:**
- Make sure you're uploading actual PDF files
- Check file size (very large files might timeout)
- Try a small test PDF first

## Quick Test

1. **Test Backend:**
   ```bash
   curl http://127.0.0.1:8000/admin/health
   ```

2. **Test File Upload (command line):**
   ```bash
   curl -X POST http://127.0.0.1:8000/admin/upload-pdf \
     -F "file=@/path/to/test.pdf"
   ```

3. **Check Database:**
   ```bash
   sqlite3 users.db "SELECT * FROM openai_files;"
   ```

## Installation Checklist

- [ ] Installed dependencies: `pip install -r requirements.txt`
- [ ] Created `.env` file with `OPENAI_API_KEY`
- [ ] Backend server is running on port 8000
- [ ] Frontend server is running on port 8080
- [ ] Browser console shows no errors
- [ ] Backend terminal shows log messages

## Still Having Issues?

1. Check backend terminal for detailed error messages
2. Check browser console (F12) for JavaScript errors
3. Verify OpenAI API key is valid and has credits
4. Try uploading a very small PDF file first
5. Check network tab in browser DevTools to see the actual HTTP request/response

