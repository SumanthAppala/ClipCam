import os
import time
import google.generativeai as genai
from pathlib import Path
import streamlit as st

                                                              
                                                            
                                                    
                                                       

class GeminiChat:
    def __init__(self, model_name="gemini-2.5-flash"):
        self.model = genai.GenerativeModel(model_name)
        self.chat_session = None
        self.uploaded_file = None

    def upload_video(self, video_path: Path):
        """Uploads the cropped clip to Gemini for processing."""
        print(f"[Gemini] Uploading {video_path}...")
        video_file = genai.upload_file(path=video_path)
        
                             
        while video_file.state.name == "PROCESSING":
            time.sleep(1)
            video_file = genai.get_file(video_file.name)
            
        if video_file.state.name == "FAILED":
            raise ValueError("Gemini failed to process the video.")
            
        print(f"[Gemini] Video processed: {video_file.uri}")
        self.uploaded_file = video_file
        return video_file

    def start_chat(self, initial_query: str):
        """
        Starts the conversation with the uploaded video context.
        """
        if not self.uploaded_file:
            raise ValueError("No video uploaded to Gemini yet.")

                                        
        system_prompt = (
            f"You are a helpful video assistant. The user searched for '{initial_query}' "
            "and this video clip was found as the result. "
            "Confirm if the incident in the query is happening in the video. "
            "Be concise. If the user asks follow-up questions, answer based on the visual details."
        )

                                                                 
        self.chat_session = self.model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [self.uploaded_file, system_prompt]
                }
            ]
        )
        
                                                           
        response = self.chat_session.send_message("Does this video match the search query? Describe what you see briefly.")
        return response.text

    def send_message(self, text: str):
        if not self.chat_session:
            return "Session not started."
        response = self.chat_session.send_message(text)
        return response.text