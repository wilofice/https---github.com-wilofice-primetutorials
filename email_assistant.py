import streamlit as st
import random
import time
from corellm import response_generator

st.title("AI Email Assistant")

# Initialize chat history using streamlit session state handler
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app restart

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



if prompt := st.chat_input("Ask me anything about your emails ?"):

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})


    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        
        response = response_generator(prompt)
        st.write(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


