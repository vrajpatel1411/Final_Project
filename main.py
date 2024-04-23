# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import streamlit as st
from time import sleep
import locale
import Response_generator
from sqlDB import DatabaseConnection

locale.getpreferredencoding = lambda: "UTF-8"


@st.cache_resource(show_spinner=False)
def load_models():
    n = Response_generator.ResponseGenerator()
    return n




rg = load_models()


def main():
    st.set_page_config(page_title="Shopper's Genie", page_icon=":shopping_trolley:")

    st.header('Your Genie for Shopping Assistance')

    if "messages" not in st.session_state.keys():  # Initialize the chat message history
        st.session_state.messages = [
            {"role": "Shopper's Genie",
             "content": "Hello! Welcome to your one-stop shop for shopping assistance. I'm your "
                        "Shopping Genie!"}
        ]

    if prompt := st.chat_input("Your question"):  # Prompt for user input and save to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages:  # Display the prior chat messages
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # If last message is not from assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "Shopper's Genie":
        with st.chat_message("Shopper's Genie"):
            with st.spinner("Gathering the information now..."):
                response = rg.getResponse(prompt)
                st.write(response)
                message = {"role": "Shopper's Genie", "content": response}
                st.session_state.messages.append(message)


# def set_up_dataset():
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
