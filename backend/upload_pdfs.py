#!/usr/bin/env python3
"""
Simple script to upload PDFs from a local folder to OpenAI.
Usage: python upload_pdfs.py /path/to/pdf/folder
"""

import os
import sys
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import sqlite3
from datetime import datetime

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY not set in .env file")
    sys.exit(1)

client = OpenAI(api_key=OPENAI_API_KEY)
DB_PATH = "users.db"

def upload_pdf_to_openai(file_path):
    """Upload a PDF file to OpenAI."""
    try:
        with open(file_path, 'rb') as f:
            file = client.files.create(
                file=f,
                purpose="assistants"
            )
        return file.id
    except Exception as e:
        print(f"Error uploading {file_path}: {str(e)}")
        return None

def store_file_id(file_id, filename):
    """Store file_id in database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if table exists, create if not
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS openai_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id TEXT UNIQUE,
            filename TEXT,
            uploaded_at TEXT,
            purpose TEXT DEFAULT 'assistants'
        )
    """)
    
    # Insert or update
    cursor.execute("""
        INSERT OR REPLACE INTO openai_files (file_id, filename, uploaded_at, purpose)
        VALUES (?, ?, ?, ?)
    """, (file_id, filename, datetime.utcnow().isoformat(), "assistants"))
    
    conn.commit()
    conn.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python upload_pdfs.py /path/to/pdf/folder")
        sys.exit(1)
    
    folder_path = Path(sys.argv[1])
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"Error: {folder_path} is not a valid directory")
        sys.exit(1)
    
    # Find all PDF files
    pdf_files = list(folder_path.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {folder_path}")
        sys.exit(0)
    
    print(f"Found {len(pdf_files)} PDF file(s)")
    
    # Delete existing files first (optional - comment out if you want to keep existing)
    print("\nDeleting existing files from OpenAI and database...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT file_id FROM openai_files")
    existing_files = cursor.fetchall()
    for (file_id,) in existing_files:
        try:
            client.files.delete(file_id)
            print(f"Deleted file: {file_id}")
        except:
            pass
    cursor.execute("DELETE FROM openai_files")
    conn.commit()
    conn.close()
    
    # Upload new files
    print("\nUploading new files...")
    uploaded_count = 0
    for pdf_file in pdf_files:
        print(f"Uploading {pdf_file.name}...")
        file_id = upload_pdf_to_openai(pdf_file)
        if file_id:
            store_file_id(file_id, pdf_file.name)
            print(f"  ✓ Uploaded successfully. File ID: {file_id}")
            uploaded_count += 1
        else:
            print(f"  ✗ Failed to upload")
    
    print(f"\n✓ Upload complete! {uploaded_count}/{len(pdf_files)} files uploaded successfully.")

if __name__ == "__main__":
    main()

