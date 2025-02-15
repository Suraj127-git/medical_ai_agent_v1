import os
from crewai import LLM

class LLMService:
    def __init__(self):
        HF_API_KEY = os.environ.get("HUGGINGFACE_API_KEY")
        self.llm = LLM(
            base_url=f"https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-alpha",
            model="huggingface/HuggingFaceH4/zephyr-7b-alpha",
            temperature=0.5,
            api_key=HF_API_KEY,
            headers={
                "Authorization": f"Bearer {HF_API_KEY}",
                "Content-Type": "application/json"
            }
        )