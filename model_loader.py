from openai import OpenAI
from config import (
    OPENROUTER_API_KEY, 
    OPENROUTER_MODEL, 
    OPENROUTER_BASE_URL,
    SITE_URL,
    SITE_NAME
)
import streamlit as st

class LLMManager:
    def __init__(self):
        self.client = None
        self.model_ready = False
        self.model_name = OPENROUTER_MODEL
        
    def initialize_client(self):
        """Initialize OpenRouter client"""
        try:
            if not OPENROUTER_API_KEY:
                raise ValueError("OpenRouter API key not found. Please set OPENROUTER_API_KEY environment variable.")
            
            self.client = OpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=OPENROUTER_API_KEY,
            )
            
            self.model_ready = True
            print(f"✅ OpenRouter client initialized with model: {self.model_name}")
            
        except Exception as e:
            raise Exception(f"Failed to initialize OpenRouter client: {str(e)}")
    
    def generate_response(self, prompt, max_tokens=1000, temperature=0.7):
        """Generate response using OpenRouter API"""
        try:
            if not self.model_ready:
                self.initialize_client()
            
            # Create the completion request
            completion = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": SITE_URL,
                    "X-Title": SITE_NAME,
                },
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant that answers questions based on provided context. Be concise, accurate, and cite relevant information from the context when possible."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            response = completion.choices[0].message.content
            return response.strip()
            
        except Exception as e:
            if "api key" in str(e).lower():
                return "❌ Error: Invalid or missing OpenRouter API key. Please check your configuration."
            elif "rate limit" in str(e).lower():
                return "⏳ Rate limit exceeded. Please wait a moment and try again."
            elif "quota" in str(e).lower():
                return "💳 Quota exceeded. Please check your OpenRouter account."
            else:
                return f"❌ Error generating response: {str(e)}"
    
    def test_connection(self):
        """Test the OpenRouter connection"""
        try:
            if not self.model_ready:
                self.initialize_client()
            
            test_response = self.generate_response("Hello, please respond with 'Connection successful!'", max_tokens=50)
            return "connection successful" in test_response.lower(), test_response
            
        except Exception as e:
            return False, str(e)

# Global instance
llm_manager = LLMManager()

def generate_response(prompt, max_tokens=1000, temperature=0.7):
    """Wrapper function for backward compatibility"""
    return llm_manager.generate_response(prompt, max_tokens, temperature)

def test_openrouter_connection():
    """Test OpenRouter connection"""
    return llm_manager.test_connection()