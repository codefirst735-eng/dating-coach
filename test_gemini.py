import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("GEMINI_API_KEY")
print(f"API Key loaded: {bool(api_key)}")
print(f"API Key length: {len(api_key) if api_key else 0}")
print(f"API Key (first 10 chars): {api_key[:10] if api_key else 'None'}...")

# Configure Gemini
genai.configure(api_key=api_key)

# Test with a simple request
try:
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content("Hello, respond with 'API is working'")
    print(f"\n✅ SUCCESS!")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"\n❌ ERROR!")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
