import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=key)

print("[*] Модели для поиска (эмбеддинги):")
for m in genai.list_models():
    if "embedContent" in m.supported_generation_methods:
        print(f"- {m.name}")