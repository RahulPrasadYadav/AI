import os
from google import genai

import dotenv
dotenv.load_dotenv()


# ENV se API key lena
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

prompt = "india capital?"

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=prompt
)

print("Gemini Response:")
print(response.text)
