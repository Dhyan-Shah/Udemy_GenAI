import streamlit as st
from langchain_groq import ChatGroq
from langchain_classic.chains import LLMMathChain,LLMChain
from langchain_classic.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_classic.callbacks import StreamlitCallbackHandler
from langchain_classic.agents import AgentType,Tool,initialize_agent

st.set_page_config(page_title='Text to Math problem solver and data search assistant',page_icon="🧮")
st.title('Text to Math problem solver using Google Gemma 2')

groq_api_key=st.sidebar.text_input(label='GROQ_API_KEY',type='password')

if not groq_api_key:
    st.info('Please enter api key to continue')
    st.stop()

llm=ChatGroq(model='openai/gpt-oss-120b',groq_api_key=groq_api_key)

wikipedia_wrapper=WikipediaAPIWrapper()
wikipedia_tool=Tool(
    name='Wikipedia',
    func=wikipedia_wrapper.run,
    description='Tool for searching the internet to find various info on the topics mentioned '
)

math_chain=LLMMathChain.from_llm(llm=llm)
calculator=Tool(
    name='Calculator',
    func=math_chain.run,
    description='A tool for answering math related questions. Only input mathematical expression need to be provided'
)

prompt="""
Your a agent tasked for solving users mathemtical question. Logically arrive at the solution and provide a detailed explanation
and display it point wise for the question below
Question:{question}
Answer:
"""

prompt_template=PromptTemplate(
    input_variables=['question'],
    template=prompt
)

chain=LLMChain(llm=llm,prompt=prompt_template)

reasoning_tool=Tool(
    name='Reasoning Tool',
    func=chain.run,
    description='A tool for answering logic-based and reasoning questions'
)

assistant_agent=initialize_agent(
    tools=[wikipedia_tool,calculator,reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handling_parsing_error=True
)

if 'messages' not in st.session_state:
    st.session_state['messages']=[
        {"role":"assistant","content":"Hi, I'm a MAth chatbot who can answer all your maths questions"}
    ]
for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg['content'])
def generate_response(question):
    response=assistant_agent.invoke({'input':question})
    return response

## LEts start the interaction
question=st.text_area("Enter youe question:","I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. Each pack of blueberries contains 25 berries. How many total pieces of fruit do I have at the end?")

if st.button('Find my answer'):
    if question:
        with st.spinner('Generate response.....'):
            st.session_state.messages.append({'role':'user','content':'question'})
            st.chat_message('user').write(question)

            st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response=assistant_agent.invoke(st.session_state.messages,callbacks=[st_cb])
            st.session_state.messages.append({'role':'assistant',"content":response})
            st.write('### Response:')
            st.success(response)
    else:
                st.warning("Please enter the question")
