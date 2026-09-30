import os
import streamlit as st

from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain_community.utilities import (
    ArxivAPIWrapper,
    WikipediaAPIWrapper
)

from langchain_community.tools import (
    ArxivQueryRun,
    WikipediaQueryRun,
    DuckDuckGoSearchRun
)

from langchain.agents import create_agent

from langchain_community.callbacks import StreamlitCallbackHandler


# ============================================================
# Load environment variables
# ============================================================

load_dotenv()


# ============================================================
# Streamlit Page
# ============================================================

st.set_page_config(
    page_title="LangChain Search Agent",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 LangChain - Chat with Search")

st.write(
    "This chatbot can search Wikipedia, Arxiv and the web "
    "using LangChain tools."
)


# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("Settings")

api_key = st.sidebar.text_input(
    "Enter your Groq API Key:",
    type="password"
)


# ============================================================
# Create Tools
# ============================================================

# -------------------------
# Wikipedia
# -------------------------

api_wrapper_wiki = WikipediaAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=250
)

wiki = WikipediaQueryRun(
    api_wrapper=api_wrapper_wiki
)


# -------------------------
# Arxiv
# -------------------------

api_wrapper_arxiv = ArxivAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=250
)

arxiv = ArxivQueryRun(
    api_wrapper=api_wrapper_arxiv
)


# -------------------------
# DuckDuckGo Search
# -------------------------

search = DuckDuckGoSearchRun(
    name="Search"
)


# List of tools
tools = [
    search,
    arxiv,
    wiki
]


# ============================================================
# Chat History
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! I'm a chatbot that can search the web, "
                "Wikipedia and Arxiv. How can I help you?"
            )
        }
    ]


# Display previous messages

for msg in st.session_state.messages:

    st.chat_message(
        msg["role"]
    ).write(
        msg["content"]
    )


# ============================================================
# User Input
# ============================================================

if prompt := st.chat_input(
    placeholder="What is machine learning?"
):

    # Add user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    # Display user message
    st.chat_message("user").write(prompt)


    # ========================================================
    # Check API Key
    # ========================================================

    if not api_key:

        st.error(
            "Please enter your Groq API key in the sidebar."
        )

        st.stop()


    # ========================================================
    # Create Groq LLM
    # ========================================================

    llm = ChatGroq(
        api_key=api_key,
        model="openai/gpt-oss-20b",
        temperature=0,
        streaming=True
    )


    # ========================================================
    # Create Agent
    # ========================================================

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are a helpful research assistant. "
            "Use the available tools whenever they are useful. "
            "You can search the web, Wikipedia and Arxiv. "
            "Give accurate and concise answers."
        )
    )


    # ========================================================
    # Run Agent
    # ========================================================

    with st.chat_message("assistant"):

        # Streamlit callback handler
        st_cb = StreamlitCallbackHandler(
            st.container(),
            expand_new_thoughts=False
        )

        try:

            response = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                },
                config={
                    "callbacks": [st_cb]
                }
            )


            # Get final AI message
            assistant_message = response[
                "messages"
            ][-1].content


            # Save assistant response
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": assistant_message
                }
            )


            # Display response
            st.write(assistant_message)


        except Exception as e:

            st.error(
                f"Error: {str(e)}"
            )