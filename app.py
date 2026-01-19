import streamlit as st
from src.chatbot import NLPBot

st.set_page_config(page_title="ADHD Support AI", page_icon="🧠")

@st.cache_resource
def load_bot():
    return NLPBot('data/questions.csv')

bot = load_bot()

st.title("🧠 ADHD Knowledge Assistant")
st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me about ADHD..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = bot.get_response(prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})