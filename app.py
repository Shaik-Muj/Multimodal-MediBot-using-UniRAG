import streamlit as st
import os
import sys
import torch
from PIL import Image

# --- Path Setup ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from src.bot import UniRAGBot 

# --- Page Configuration ---
st.set_page_config(
    page_title="UniRAG Conversational Bot",
    page_icon="🩺",
    layout="wide"
)

# --- Constants ---
TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

# --- Load the Bot (The "Brain") ---
@st.cache_resource
def load_bot():
    print("Loading UniRAG Conversational Agent... (This happens only once)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bot = UniRAGBot(device=device)
    return bot

# Load the bot and its chat history (which is now stored *inside* the bot)
bot = load_bot()

# --- Sidebar for Image Uploads ---
with st.sidebar:
    st.title("🩺 UniRAG Medical Bot")
    st.markdown("This bot is now a conversational agent. Ask follow-up questions!")
    
    st.divider()
    
    st.subheader("Hybrid Image Query")
    uploaded_file = st.file_uploader("Upload a medical scan (JPG, PNG)", type=["jpg", "png", "jpeg"])
    
    text_hint = st.text_input(
        "Optional: Add a text description",
        placeholder="e.g., 'brain mri scan'"
    )
    
    if st.button("Analyze Image") and uploaded_file:
        temp_path = os.path.join(TEMP_DIR, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.image(uploaded_file, caption="Uploaded Image")
        
        with st.spinner("Analyzing image with hybrid search..."):
            image_response = bot.image_query(temp_path, text_hint if text_hint else None)
        
        os.remove(temp_path)
        # We don't need to manually manage session_state, 
        # the bot's internal history is the source of truth.

# --- Main Chat Interface ---
st.title("Chat with your Medical Assistant")

# Display the chat history from *inside the bot*
for role, message in bot.chat_history:
    with st.chat_message(role):
        st.markdown(message)

# Get new user input
if prompt := st.chat_input("What are your symptoms?"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Get bot response (which now also updates the bot's internal history)
    with st.spinner("Thinking..."):
        bot_response = bot.text_query(prompt)
        
    # Display bot response
    with st.chat_message("assistant"):
        st.markdown(bot_response)
        
    # Force Streamlit to re-run and display the new history
    st.rerun()