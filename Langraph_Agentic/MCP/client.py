from langchain_mcp_adapters import MultiServerMCPClient, client, client
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import asyncio
import os
load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

async def main():
    client=MultiServerMCPClient(
        {
            'math':{
                'command':'python',
            'args':['MCP/mathserver.py'],
            'transport':'stdio'  
            },
            'weather':{
                'command':'python',
                'args':['MCP/weather.py'],
                'transport':'streamable-http'
            }
        }
    )
    tools=await client.get_tools()
    # from langchain_groq import ChatGroq

    # model=ChatGroq(model='openai/gpt-oss-20b')
    model=ChatGroq(model="openai/gpt-oss-20b")
    agent=create_react_agent(model,tools)

    math_response=await agent.arun("What is 5 + 3?")
    print(f"Math Response: {math_response}")
    weather_response=await agent.arun("What is the weather in New York?")
    print(f"Weather Response: {weather_response}")

asyncio.run(main())
