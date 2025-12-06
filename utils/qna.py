# utils/qna.py
import os
from google import genai

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def refine_query(query: str) -> str:
    response = client.models.generate_content(
        model="gemini-2.5-flash",       # or whichever model your key supports
        contents=f"Refine this search query for video search: {query}"
    )
    return response.text.strip()

def answer_question(history, question: str, video_summary: str) -> str:
    chat_context = ""
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        chat_context += f"{role}: {msg['content']}\n"

    prompt = (
        "You are a helpful AI assistant skilled in interpreting video search results.\n"
        f"{chat_context}\n"
        f"Video Summary: {video_summary}\n"
        f"Question: {question}\n"
        "Answer concisely."
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text.strip()
