from google import genai
import os
from dotenv import load_dotenv
# The client gets the API key from the environment variable `GEMINI_API_KEY`.
load_dotenv()
GENAI_API_KEY = os.getenv("GENAI_API_KEY")
client = genai.Client(api_key=GENAI_API_KEY)

response = client.models.generate_content(
    model="gemini-2.5-flash", contents="Explain how AI works in a few words, 20 words"
)
print(response.text)