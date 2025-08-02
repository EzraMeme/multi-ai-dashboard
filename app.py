import os
import streamlit as st
import requests
from dotenv import load_dotenv
from anthropic import Anthropic
from google.generativeai import GenerativeModel, configure
from openai import OpenAI

# Load environment variables
load_dotenv()

# Set up Streamlit UI
st.set_page_config(page_title="AI Comparison Dashboard", layout="wide")
st.title("📊 Compare Responses Across Top Free AI Models")

# Prompt input
prompt = st.text_area("Enter your prompt:", height=200)

# Load API keys
openai_api_key = os.getenv("OPENAI_API_KEY", "")
claude_api_key = os.getenv("ANTHROPIC_API_KEY", "")
gemini_api_key = os.getenv("GEMINI_API_KEY", "")
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY", "")
groq_api_key = os.getenv("GROQ_API_KEY", "")

# Helper functions

def run_openai(prompt):
    try:
        client = OpenAI(api_key=openai_api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ OpenAI error: {str(e)}"

def run_claude(prompt):
    try:
        client = Anthropic(api_key=claude_api_key)
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1000,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    except Exception as e:
        return f"❌ Claude error: {str(e)}"

def run_gemini(prompt):
    try:
        configure(api_key=gemini_api_key)
        model = GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Gemini Error: {str(e)}"

def run_huggingface(prompt):
    try:
        headers = {
            "Authorization": f"Bearer {huggingface_api_key}"
        }
        payload = {
            "inputs": prompt,
            "parameters": {"temperature": 0.7}
        }
        url = "https://api-inference.huggingface.co/models/google/flan-t5-xl"
        response = requests.post(url, headers=headers, json=payload)
        result = response.json()
        return result[0]["generated_text"] if isinstance(result, list) else str(result)
    except Exception as e:
        return f"❌ HuggingFace error: {str(e)}"

def run_groq_llama3(prompt):
    try:
        client = OpenAI(
            api_key=groq_api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Groq (LLaMA3) error: {str(e)}"

# Run models and display results
if st.button("Run Prompt"):
    with st.spinner("Querying models..."):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("🧠 OpenAI (ChatGPT)")
            st.markdown(run_openai(prompt))

        with col2:
            st.subheader("🧠 Claude")
            st.markdown(run_claude(prompt))

        with col3:
            st.subheader("🧠 Gemini")
            st.markdown(run_gemini(prompt))

        col4, col5 = st.columns(2)

        with col4:
            st.subheader("🧠 HuggingFace")
            st.markdown(run_huggingface(prompt))

        with col5:
            st.subheader("🧠 Groq (LLaMA3)")
            st.markdown(run_groq_llama3(prompt))