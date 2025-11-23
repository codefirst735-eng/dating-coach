# OpenAI File Search Integration

This backend now supports OpenAI's File Search/Retrieval system for enhanced AI coaching responses.

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set OpenAI API Key**
   - Create a `.env` file in the project root
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```
   - Get your API key from: https://platform.openai.com/api-keys

3. **Start the Server**
   ```bash
   uvicorn main:app --reload
   ```

## Features

### 1. PDF Upload
- Upload PDF files via the admin panel at `/admin.html`
- Files are uploaded to OpenAI with `purpose="assistants"`
- File IDs are stored in the database

### 2. File Management
- **List Files**: `GET /admin/list-files`
- **Upload File**: `POST /admin/upload-pdf`
- **Delete File**: `DELETE /admin/delete-file/{file_id}`
- **Replace All**: `POST /admin/replace-all-files`

### 3. Enhanced Chat
- The `/chat` endpoint now uses OpenAI's Assistants API with file search
- System prompt: "You are a direct, confident, redpill no bullshit dating coach..."
- PDFs are automatically used as knowledge sources for responses

## Admin Panel

Access the admin panel at: `http://localhost:8080/admin.html`

Features:
- Upload multiple PDFs
- View all uploaded files
- Delete individual files
- Replace all files at once

## How It Works

1. **File Upload**: PDFs are uploaded to OpenAI and file IDs are stored in the database
2. **Vector Store**: When files exist, a vector store is created with all file IDs
3. **Assistant**: An assistant is created with file_search tool pointing to the vector store
4. **Chat**: User messages are processed through the assistant, which searches the PDFs for relevant context

## Notes

- Only PDF files are supported
- Files are stored in OpenAI's system (not locally)
- The assistant is recreated for each chat session (can be optimized to cache)
- If OpenAI API is not configured, the system falls back to random responses

