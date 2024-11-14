import streamlit as st
from transformers import pipeline
from datetime import datetime
import re
import random

# Set page config first
st.set_page_config(page_title="Chat Interface", layout="wide")

# Custom CSS for better chat interface
st.markdown("""
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e;
    margin-left: 20%;
    border-bottom-right-radius: 0.125rem;
}
.chat-message.assistant {
    background-color: #475063;
    margin-right: 20%;
    border-bottom-left-radius: 0.125rem;
}
.chat-message .avatar {
    width: 20%;
}
.chat-message .message {
    width: 80%;
    padding: 0 1.5rem;
    color: #fff;
}
.chat-message .timestamp {
    font-size: 0.8rem;
    color: #8c8c8c;
}
.stTextInput {
    position: fixed;
    bottom: 3rem;
    background-color: white;
    padding: 1rem;
    width: 80%;
}
.main {
    max-width: 800px;
    margin: auto;
}
</style>
""", unsafe_allow_html=True)

# Load the text generation model
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="microsoft/DialoGPT-medium")

# Initialize the model
generator = load_model()

# Predefined responses for common greetings
GREETING_PATTERNS = {
    r'\b(hi|hello|hey)\b': [
        "Hello! How can I help you today?",
        "Hi there! What's on your mind?",
        "Hey! Nice to meet you.",
    ],
    r'\bhow are you\b': [
        "I'm doing well, thank you for asking! How are you?",
        "I'm great! How can I assist you today?",
        "I'm functioning perfectly! What can I help you with?",
    ],
    r'\bgood morning\b': [
        "Good morning! How can I make your day better?",
        "Good morning! I hope you're having a great start to your day.",
    ],
    r'\bgood (evening|night)\b': [
        "Good evening! How can I assist you?",
        "Good evening! What can I help you with?",
    ],
}

def generate_response(prompt):
    # Check for greeting patterns first
    for pattern, responses in GREETING_PATTERNS.items():
        if re.search(pattern, prompt.lower()):
            return random.choice(responses)
    
    # If no greeting pattern matches, generate response using the model
    response = generator(
        prompt,
        max_length=50,
        num_return_sequences=1,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        no_repeat_ngram_size=2,
        pad_token_id=50256
    )
    response_text = response[0]['generated_text']
    
    # Clean up response
    if response_text.startswith(prompt):
        response_text = response_text[len(prompt):].strip()
    
    # Remove repeated sentences
    sentences = re.split(r'(?<=[.!?])\s+', response_text)
    unique_sentences = []
    for sentence in sentences:
        if sentence and sentence not in unique_sentences:
            unique_sentences.append(sentence)
    
    response_text = ' '.join(unique_sentences)
    
    # If response is empty or too short, provide a fallback response
    if len(response_text.strip()) < 2:
        return "I understand. Please tell me more about that."
    
    return response_text.strip()

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Initialize input key
if 'input_key' not in st.session_state:
    st.session_state.input_key = 0

# Main container
with st.container():
    # Chat title
    st.title("💬 Chat Interface")
    st.markdown("---")

    # Chat messages container
    chat_container = st.container()

    # Input container
    with st.container():
        # Create two columns for input and button
        col1, col2 = st.columns([6,1])
        
        with col1:
            user_input = st.text_input("", placeholder="Type your message here...", key=f"user_input_{st.session_state.input_key}")
        
        with col2:
            send_button = st.button("Send")

    # Handle send button click
    if send_button and user_input:
        # Add timestamp
        timestamp = datetime.now().strftime("%H:%M")
        
        # Add user message to chat history
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": timestamp
        })

        # Generate bot response
        response_text = generate_response(user_input)

        # Add bot response to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "timestamp": datetime.now().strftime("%H:%M")
        })

        # Clear input by incrementing the key
        st.session_state.input_key += 1
        st.rerun()

    # Display chat messages
    with chat_container:
        for message in st.session_state.messages:
            message_type = message["role"]
            message_content = message["content"]
            timestamp = message["timestamp"]

            # Create message container
            st.markdown(f"""
                <div class="chat-message {message_type}">
                    <div class="message">
                        {message_content}
                        <div class="timestamp">{timestamp}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    # Add some space at the bottom
    st.markdown("<br>" * 5, unsafe_allow_html=True)

# Clear chat button (in sidebar)
with st.sidebar:
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()