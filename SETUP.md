# 🚀 RFH - Setup Guide for New Machine

This guide will help you set up the Relationship for Humans (RFH) project on a new laptop.

## Prerequisites

Before you begin, ensure you have the following installed:

1. **Git** - [Download](https://git-scm.com/downloads)
2. **Python 3.8+** - [Download](https://www.python.org/downloads/)
3. **Node.js** (optional, for package management) - [Download](https://nodejs.org/)

---

## 📦 Step 1: Clone the Repository

```bash
# Navigate to your desired directory
cd ~/Desktop/coding_projects

# Clone the repository
git clone https://github.com/codefirst735-eng/dating-coach.git

# Navigate into the project
cd dating-coach

# Switch to development branch
git checkout development
```

---

## 🐍 Step 2: Set Up Python Virtual Environment

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install Python dependencies
pip install -r backend/requirements.txt
```

---

## 🔑 Step 3: Configure Environment Variables

Create a `.env` file in the project root:

```bash
touch .env
```

Add the following content to `.env`:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

**Important:**
- Get your Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
- The `.env` file is already in `.gitignore`, so it won't be committed to the repository

---

## 🗄️ Step 4: Initialize the Database

The SQLite database will be created automatically when you first run the backend. No manual setup is required.

---

## ▶️ Step 5: Run the Application

You'll need **TWO** terminal windows:

### Terminal 1: Backend Server

```bash
# Make sure you're in the project root directory
cd ~/Desktop/coding_projects/dating-coach

# Activate virtual environment (if not already activated)
source venv/bin/activate

# Start the backend server
venv/bin/python -m uvicorn backend.main:app --reload --port 8001
```

The backend will be available at: **http://127.0.0.1:8001**

### Terminal 2: Frontend Server

```bash
# Navigate to the project root (in a NEW terminal window)
cd ~/Desktop/coding_projects/dating-coach

# Start the frontend server
python3 -m http.server 8082 -d frontend
```

The frontend will be available at: **http://127.0.0.1:8082**

---

## 🌐 Step 6: Access the Application

Open your browser and go to:

```
http://127.0.0.1:8082
```

**Default Pages:**
- Home: `http://127.0.0.1:8082/index.html`
- Login: `http://127.0.0.1:8082/login.html`
- Signup: `http://127.0.0.1:8082/signup.html`
- Chat: `http://127.0.0.1:8082/chat.html`
- Admin: `http://127.0.0.1:8082/admin.html`

---

## 📚 Step 7: Upload PDF Knowledge Base (Admin Only)

1. Navigate to `http://127.0.0.1:8082/admin.html`
2. Select coach type (Male or Female)
3. Upload PDF files for the AI knowledge base
4. The AI will use these PDFs to provide contextualized advice

---

## 🔧 Troubleshooting

### Port Already in Use

If you get a "port already in use" error:

```bash
# For port 8001 (backend):
lsof -ti:8001 | xargs kill -9

# For port 8082 (frontend):
lsof -ti:8082 | xargs kill -9
```

### Module Not Found Errors

Make sure your virtual environment is activated:

```bash
source venv/bin/activate
```

Then reinstall dependencies:

```bash
pip install -r backend/requirements.txt
```

### Database Errors

If you encounter database errors, you can reset it:

```bash
rm users.db
# The database will be recreated when you restart the backend
```

### Gemini API Not Working

1. Check that your `GEMINI_API_KEY` is correctly set in `.env`
2. Verify the API key is valid at [Google AI Studio](https://makersuite.google.com/app/apikey)
3. Restart the backend server after updating the `.env` file

---

## 📁 Project Structure

```
dating-coach/
├── backend/
│   ├── main.py              # FastAPI backend
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── index.html           # Landing page
│   ├── chat.html            # AI Coach chat interface
│   ├── admin.html           # PDF management
│   ├── login.html           # Login page
│   ├── signup.html          # Registration page
│   ├── guide.html           # User guide
│   ├── terms.html           # Terms & Conditions
│   ├── privacy.html         # Privacy Policy
│   ├── components/          # Reusable components (header/footer)
│   ├── css/                 # Stylesheets
│   ├── js/                  # JavaScript files
│   └── img/                 # Images
├── .env                     # Environment variables (create this)
├── .gitignore              # Git ignore file
├── users.db                # SQLite database (auto-generated)
└── SETUP.md                # This file
```

---

## 🔄 Keeping Your Code Updated

### Pull Latest Changes

```bash
git pull origin development
```

### Push Your Changes

```bash
# Stage all changes
git add .

# Commit with a descriptive message
git commit -m "feat: your feature description"

# Push to development branch
git push origin development
```

---

## 🎯 Key Features

1. **AI Coaching**: Separate coaches for men and women with cultural Indian context
2. **Conversation Memory**: AI remembers last 20 messages for context
3. **PDF Knowledge Base**: Upload PDFs to enhance AI responses
4. **Screenshot Analysis**: Analyze text conversations
5. **Subscription Plans**: Sleeper, Initiate, and Master tiers
6. **Guest Access**: Limited access for non-registered users

---

## 🆘 Need Help?

- Check the browser console for frontend errors (F12)
- Check the terminal for backend errors
- Review the `.env` file for correct API keys
- Ensure both servers are running

---

## 📝 Notes

- The application uses **SQLite** for the database (serverless)
- **Session timeout** is set to 24 hours
- **Local development** uses `http://127.0.0.1` URLs
- **Production deployment** requires updating URLs in `js/config.js`

---

Happy Coding! 🚀
