import os
from dotenv import load_dotenv

# 1. Load variables from the .env file into the system environment
load_dotenv()

# 2. Safely retrieve the key
api_key = os.environ.get("GROQ_API_KEY")

# 3. Test verification block
if api_key:
    print("✅ Successfully loaded GROQ_API_KEY from .env!")
else:
    print("❌ Failed to find GROQ_API_KEY. Check your .env file placement.")