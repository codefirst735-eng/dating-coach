import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ Error: GEMINI_API_KEY not found in .env file")
    exit(1)

print(f"🔑 Testing API Key: {api_key[:5]}...{api_key[-5:]}")

try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    print("📡 Sending request to Gemini...")
    response = model.generate_content("Hello, are you working?")
    
    print("\n✅ Success! Response from Gemini:")
    print(response.text)

except Exception as e:
    print("\n❌ API Error:")
    print(str(e))
