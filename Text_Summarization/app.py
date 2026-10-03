import validators
import streamlit as st

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import (
    YoutubeLoader,
    UnstructuredURLLoader
)


# ---------------------------------------------------------
# Streamlit App Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="LangChain: Summarize Text From YT or Website",
    page_icon="🦜"
)

st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader("Summarize URL")


# ---------------------------------------------------------
# Sidebar - Groq API Key
# ---------------------------------------------------------

with st.sidebar:
    groq_api_key = st.text_input(
        "Groq API Key",
        value="",
        type="password"
    )


# ---------------------------------------------------------
# URL Input
# ---------------------------------------------------------

generic_url = st.text_input(
    "URL",
    label_visibility="collapsed",
    placeholder="Enter YouTube or website URL..."
)


# ---------------------------------------------------------
# Prompt
# ---------------------------------------------------------

prompt_template = """
Provide a concise summary of the following content in approximately 300 words.

Content:
{text}
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["text"]
)


# ---------------------------------------------------------
# Summarization Button
# ---------------------------------------------------------

if st.button("Summarize the Content from YT or Website"):

    # -----------------------------------------------------
    # Validate API key and URL
    # -----------------------------------------------------

    if not groq_api_key.strip():
        st.error("Please provide your Groq API Key.")

    elif not generic_url.strip():
        st.error("Please provide a URL.")

    elif not validators.url(generic_url):
        st.error("Please enter a valid URL.")

    else:

        try:

            with st.spinner("Loading and summarizing..."):

                # -------------------------------------------------
                # Create Groq LLM
                # -------------------------------------------------

                llm = ChatGroq(
                    model="openai/gpt-oss-20b",
                    groq_api_key=groq_api_key,
                    temperature=0
                )

                # -------------------------------------------------
                # Load YouTube or Website
                # -------------------------------------------------

                if (
                    "youtube.com" in generic_url
                    or "youtu.be" in generic_url
                ):

                    loader = YoutubeLoader.from_youtube_url(
                        generic_url,
                        add_video_info=True
                    )

                else:

                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": (
                                "Mozilla/5.0 "
                                "(Windows NT 10.0; Win64; x64) "
                                "AppleWebKit/537.36 "
                                "(KHTML, like Gecko) "
                                "Chrome/116.0.0.0 "
                                "Safari/537.36"
                            )
                        }
                    )

                # -------------------------------------------------
                # Load Documents
                # -------------------------------------------------

                docs = loader.load()

                if not docs:
                    st.error("Could not extract any content from the URL.")
                    st.stop()

                # -------------------------------------------------
                # Summarization Chain
                # -------------------------------------------------

                chain = load_summarize_chain(
                    llm=llm,
                    chain_type="stuff",
                    prompt=prompt
                )

                # -------------------------------------------------
                # Run Chain
                # -------------------------------------------------

                result = chain.invoke(docs)

                # -------------------------------------------------
                # Display Result
                # -------------------------------------------------

                if isinstance(result, dict):
                    output_summary = result.get(
                        "output_text",
                        result.get("text", str(result))
                    )
                else:
                    output_summary = str(result)

                st.success("Summary generated successfully!")

                st.write(output_summary)

        except Exception as e:

            st.error("Something went wrong.")
            st.exception(e)