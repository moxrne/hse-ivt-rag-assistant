import os
from dotenv import load_dotenv
from llama_index.llms.gemini import Gemini

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

llm = Gemini(model="models/gemini-flash-latest", api_key=api_key)
print("[*] Стучимся в Gemini...")
response = llm.complete("Привет! Скажи одним предложением, кто ты?")
print("Ответ:", response)