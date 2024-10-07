import streamlit as st
# from openai import OpenAI
from utils import invoke_chain
st.title("ShikshaLokam Query Hub 🔍")

# Move database selection to the sidebar
options = ["Mentoring", "SCP", "Projects", "Katha"]
selected_option = st.sidebar.radio("Choose DB:", options)
print("++++++++++++\n" + selected_option)

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Initialize chat history
if "messages" not in st.session_state:
    # print("Creating session state")
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):


    print("Prompt:", prompt)
    print("Selected Option:", selected_option)
    print("Session State Messages:", st.session_state.messages)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.spinner("Generating response..."):
        with st.chat_message("assistant"):
            response = invoke_chain(prompt,st.session_state.messages,selected_option)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})