from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from langserve import add_routes
from dotenv import load_dotenv
load_dotenv()

groq_api_key=os.getenv("GROQ_API_KEY")
model=ChatGroq(model='llama-3.1-8b-instant', api_key=groq_api_key, temperature=0.1)


# 1. cREATE PROMPT TEMPLATE
generic_template='Translate the following into {language}:'
prompt=ChatPromptTemplate.from_messages([
    ('system',generic_template),
    ('user','{text}')
])


# 2. cREATE OUTPUT PARSER
parser=StrOutputParser()


# 3. cREATE CHAIN
chain=prompt|model|parser


## App definition
app=FastAPI(title="LCEL Example", description="A simple example of using LCEL to create a translation service.", version="1.0"    )

add_routes(
    app,chain,path='/chain'
)

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)