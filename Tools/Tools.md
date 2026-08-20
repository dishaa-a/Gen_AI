# Tools
In LangChain, tools are predefined functions, APIs, or computational modules that extend the capabilities of Large Language Models (LLMs) by allowing them to perform external actions like data retrieval, calculations, or API calls.  

# Agents 
Agents are LLM-powered entities that reason, plan, and decide which tools to invoke to solve complex, multi-step queries. 
An AI agent is an LLM powered system that can autonomously think, decide, and take actions using external tools or APIs to achieve a goal.
Agents are more intelligent than a standalone LLM because they can:

Select tools based on task requirements.
Chain multiple steps.
Observe outputs and adjust decisions dynamically.

# Core Components and Types
Tool Definition: Tools can be simple Python functions wrapped with the @tool decorator, StructuredTool for complex inputs, or built-in classes like WikipediaQueryRun or DuckDuckGoSearchResults. 

Agent Types: Common patterns include Tool Calling Agents for direct function invocation, ReAct Agents for reasoning and action steps, and Structured Chat Agents for JSON-formatted outputs. 

Prebuilt Toolkits: LangChain provides toolkits for various services, including Wikipedia, Google Search, SQL databases, GitHub, Gmail, and Financial Datasets. 

In LangChain, tools are functions that an LLM/agent can call to perform actions beyond just generating text.

There are two main ways to use them:

1. Built-in tools — already provided by LangChain or its integrations.
2. Custom tools — functions you create yourself.

# 1. Built-in LangChain Tools
LangChain provides tools for common tasks such as:

DuckDuckGoSearchRun - Web search via DuckDuckGo
WikipediaQueryRun - Wikipedia summary
PythonREPLTool - Run raw Python code
ShellTool - Run shell commands
RequestsGetTool - Make HTTP GET requests
GmailSendMessageTool - Sends emails via Gmail
SlackSendMessageTool - Post message to Slack
SQLDatabaseQueryTool - Run SQL Queries

# 2. Custom Tools

A custom tool is a Python function that you turn into a LangChain tool.

Use them when:
1. You want to call your own APIs
2. You want to encapsulate business logic
3. You want the LLM to interact with your database, product or app

The easiest modern approach is the @tool decorator:

from langchain_core.tools import tool

@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

print(add_numbers.invoke({"a": 10, "b": 20}))

==> Tool with Arguments
from langchain_core.tools import tool

@tool
def search_student(name: str) -> str:
    """Search for information about a student by name."""
    
    students = {
        "Disha": "B.Tech AI/ML student",
        "Chetna": "B.Tech CSE student"
    }

    return students.get(name, "Student not found")

search_student.invoke({"name": "Disha"})

# Toolkits
A toolkit in LangChain is simply a collection of related tools packaged together so an agent can use them.

Tool
  ↓
One capability

Toolkit
  ↓
Collection of related tools

Agent
  ↓
Uses one or more tools from the toolkit

# Tool Binding
Tool binding means connecting one or more tools to an LLM so that the LLM knows which tools are available and can request their use.

In simple terms:

Tool binding = giving tools to the LLM.

# Tool calling
Tool calling is when an LLM decides that it needs to use a specific tool and returns a structured request telling your program which tool to execute and with what arguments.

The important distinction is:

The LLM usually does not execute the Python function itself. It asks your application to execute it.

import os
from huggingface_hub import login
from dotenv import load_dotenv

load dotenv()

HF_TOKEN = os.getenv("huggingfacehub_api_token")
login(token = HF_TOKEN)

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace, HuggingFacePipeline
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests

# tool create

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a*b

print(multiply.invoke({"a": 2, "b": 6}))

# tool binding
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini"
)

llm_with_tools = llm.bind_tools([multiply])
llm_with_tools.invoke('Hi how are you')
result = llm_with_tools.invoke('Can you multiply 2 with 4')
result.tool_calls[0]['args']
multiply.invoke(result.tool_calls[0]['args'])

# Tool execution
Tool execution means actually running the tool/function after the LLM has requested it.

This is the next step after tool calling.
