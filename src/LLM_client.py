import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

class HuggingFaceChatLLM:
    def __init__(self, model_id="mistralai/Mistral-7B-Instruct-v0.3", temperature=0):
        token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not token:
            raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in environment variables.")
        self.model_id = model_id
        self.temperature = temperature
        self.client = InferenceClient(token=token)

    def invoke(self, prompt, **kwargs):
        if not isinstance(prompt, str):
            raise ValueError("Prompt must be a string.")

        messages = [
            {"role": "system", "content": "You are an ADGM compliance expert."},
            {"role": "user", "content": prompt}
        ]
        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=self.temperature,
            max_tokens=kwargs.get("max_tokens", 1000)
        )
        return resp.choices[0].message["content"]

def get_LLM():
    return HuggingFaceChatLLM()
