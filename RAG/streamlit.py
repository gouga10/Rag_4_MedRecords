import streamlit as st
import requests
def get_last_4_messages(conversation):
    # Take the last 4 messages (if available)
    last_msgs = conversation[-4:]

    # Initialize an empty string for the formatted conversation
    formatted_conversation = ""

    # Iterate through the last messages and format them
    for msg in last_msgs:
        role = msg['role']
        content = msg['content']
        formatted_conversation += f"{role}: {content}\n"
    
    return formatted_conversation.strip()
# Title of the app
st.title("Medical Records Chatbot")

# Initialize session state to store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Send user message to local API and get response
    api_url = "http://localhost:8001/generate"
    payload = {"query": prompt,'conv':get_last_4_messages(st.session_state.messages)}
    response = requests.post(api_url, json=payload)

    if response.status_code == 200:
        bot_response = response.json().get("response", "Sorry, I couldn't understand that.")
    else:
        bot_response = "Sorry, there was an error processing your request."

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(bot_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": bot_response})