import streamlit as st
import os
import sys
import torch 
from PIL import Image

# --- Path Setup ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from src.bot import MediRAGBot 

# --- Page Configuration ---
st.set_page_config(
    page_title="UniRAG Medical",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- UI CONSTANTS ---
BOT_AVATAR = "🤖" 
USER_AVATAR = "👤"

# --- 🎨 CUSTOM CSS OVERHAUL ---
st.markdown("""
<style>
    /* --- 1. RESET & MAIN BACKGROUND --- */
    .stApp {
        background-color: #212121; /* Deep Grey/Black Main BG */
        color: #ECECEC;
        font-family: 'Inter', sans-serif;
    }
    
    /* --- 2. SIDEBAR STYLING (Matches Screenshot) --- */
    [data-testid="stSidebar"] {
        background-color: #171717; /* Darker Sidebar */
        border-right: 1px solid #333333;
    }
    
    /* Custom "New Chat" Button Styling in Sidebar */
    div.stButton > button {
        width: 100%;
        background-color: transparent;
        border: 1px solid #444;
        color: #ECECEC;
        border-radius: 8px;
        text-align: left;
        padding: 0.5rem 1rem;
        transition: all 0.2s;
    }
    div.stButton > button:hover {
        background-color: #2A2A2A;
        border-color: #666;
        color: #FFF;
    }
    
    /* --- 3. CHAT MESSAGE BUBBLES (The Contrast You Asked For) --- */
    
    /* USER MESSAGE: Purple/Blue Gradient */
    [data-testid="stChatMessage"][data-testid*="user"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 18px 18px 0px 18px; /* Rounded with a sharp corner */
        padding: 15px;
        margin-bottom: 15px;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* BOT MESSAGE: Dark Glassmorphism Card */
    [data-testid="stChatMessage"][data-testid*="assistant"] {
        background-color: #2F2F2F;
        color: #E0E0E0;
        border-radius: 18px 18px 18px 0px;
        padding: 15px;
        margin-bottom: 15px;
        border: 1px solid #3A3A3A;
    }
    
    /* Hide the default avatars inside the message container to use our own style if needed, 
       but keeping them enabled for clarity. We style the avatar container below. */
    [data-testid="chatAvatarIcon"] {
        background-color: transparent;
    }

    /* --- 4. FILE UPLOADER (Dashed Box Style) --- */
    [data-testid="stFileUploader"] {
        background-color: #1E1E1E;
        border: 1px dashed #444;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    [data-testid="stFileUploader"] section {
        background-color: transparent;
    }
    
    /* --- 5. CHAT INPUT (Floating Pill) --- */
    .stChatInputContainer {
        background-color: #212121; /* Matches main bg */
        padding-bottom: 30px;
        padding-top: 10px;
    }
    .stChatInput textarea {
        background-color: #2F2F2F;
        color: white;
        border: 1px solid #444;
        border-radius: 24px; /* Pill shape */
        padding: 15px 25px;
    }
    .stChatInput textarea:focus {
        border-color: #764ba2; /* Purple glow focus */
        box-shadow: 0 0 0 1px #764ba2;
    }
    
    /* --- 6. HERO SECTION (Watermark Look) --- */
    .hero-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 70vh;
        color: #444; /* Faint Text */
        user-select: none;
    }
    .hero-title {
        font-size: 80px;
        font-weight: 800;
        color: #2A2A2A; /* Very faint watermark style */
        margin: 0;
    }
    .hero-subtitle {
        font-size: 20px;
        color: #3A3A3A;
        margin-top: -10px;
    }

    /* Hide header/footer */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
</style>
""", unsafe_allow_html=True)

# --- Constants ---
TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

# --- Load Bot ---
@st.cache_resource
def load_bot():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return MediRAGBot(device=device)

try:
    bot = load_bot()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
#               SIDEBAR
# ==========================================
with st.sidebar:
    # Title Header
    st.markdown("### 🏥 UniRAG Medical")
    
    # New Chat Button (Styled as Outline Button via CSS)
    if st.button("＋ New Chat"):
        bot.chat_history = []
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Upload Section
    st.markdown("**Upload Scan**")
    uploaded_file = st.file_uploader(
        "Drag & drop file here", 
        type=["jpg", "png", "jpeg"],
        label_visibility="collapsed"
    )
    
    text_hint = None
    if uploaded_file:
        st.success(f"Loaded: {uploaded_file.name}")
        text_hint = st.text_input("Context (Optional)", placeholder="e.g. 'Brain MRI'")
        
        # Analyze Button (Primary Action)
        if st.button("🚀 Analyze Scan", type="primary"):
            temp_path = os.path.join(TEMP_DIR, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # User Message
            user_msg = f"**Analysis Request:** `{uploaded_file.name}`"
            if text_hint: user_msg += f"\n\n*Context: {text_hint}*"
            st.session_state.messages.append({"role": "user", "content": user_msg})
            
            # Render Image in Chat
            with st.chat_message("user", avatar=USER_AVATAR):
                st.image(uploaded_file, width=350)
                st.markdown(user_msg)

            # Bot Response (Stream)
            with st.chat_message("assistant", avatar=BOT_AVATAR):
                response = st.write_stream(bot.image_query_stream(temp_path, text_hint))
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
            st.rerun()

    # Bottom Menu Simulation
    st.markdown("<br>" * 10, unsafe_allow_html=True) # Spacer
    st.markdown("---")
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🗑️", help="Clear History"):
            bot.chat_history = []
            st.session_state.messages = []
            st.rerun()
    with col2:
        st.caption("Clear Conversation")

# ==========================================
#               MAIN CHAT
# ==========================================

# 1. Hero Section (Empty State)
if len(st.session_state.messages) == 0:
    st.markdown("""
        <div class="hero-container">
            <div class="hero-title">UniRAG</div>
            <div class="hero-subtitle">Your Medical Assistant</div>
        </div>
    """, unsafe_allow_html=True)

# 2. Chat History
else:
    # Add a little top padding so messages don't stick to the top
    st.markdown("<div style='padding-top: 20px;'></div>", unsafe_allow_html=True)
    
    for msg in st.session_state.messages:
        avatar = BOT_AVATAR if msg["role"] == "assistant" else USER_AVATAR
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

# 3. Chat Input
if prompt := st.chat_input("Describe your symptoms or ask a question..."):
    # User
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)
    
    # Bot
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        response = st.write_stream(bot.text_query_stream(prompt))
    
    # Save
    st.session_state.messages.append({"role": "assistant", "content": response})