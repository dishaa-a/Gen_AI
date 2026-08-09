Runnables in LangChain are a unified abstraction that provides a common interface for all components (LLMs, prompts, retrievers, parsers, and custom functions), allowing them to be connected into modular, composable pipelines via the LangChain Expression Language (LCEL).

This standardization enables developers to use consistent methods like invoke(), batch(), and stream() across diverse elements without remembering separate APIs for each. 

Let's see why runnables were needed at the first place.

Files flow goes as:

simple_llm.py -> pdf_reader.py -> llmchain.py

## simple_llm.py has been coded as:
Wraps the Hugging Face pipeline into a LangChain chat model.
Defines a reusable prompt with a {topic} variable.
Gets the blog topic from the user.
Replaces {topic} with the user's input in formatted_prompt.
Sends the formatted prompt to the LLM and generates a title.
Prints the generated blog title.

User Topic → Prompt Template → Formatted Prompt → Qwen LLM → Generated Blog Title

## pdf-reader.py has been coded as(RAG):
Loads the content of docs.txt into LangChain Document objects.
Breaks the document into smaller overlapping chunks so they can be efficiently searched.
Converts the text chunks into numerical vectors (embeddings) and stores them in a FAISS vector database.
Creates a component that searches the vector database for chunks relevant to a query.
Defines the question that we want the model to answer.
Searches FAISS and retrieves the chunks that are most relevant to the question.
Combines the retrieved chunks into one piece of text.
Gives the LLM both the question and the relevant information retrieved from the document.
The LLM generates an answer based on the retrieved document content.

docs.txt
   ↓
Load Document
   ↓
Split into Chunks
   ↓
Create Embeddings
   ↓
Store in FAISS
   ↓
User Query
   ↓
Retriever
   ↓
Relevant Chunks
   ↓
Prompt + Retrieved Text
   ↓
Qwen LLM
   ↓
Answer

It introduces the important RAG concept.

## llmcchain.py has been coded as:
User enters topic
       ↓
PromptTemplate
       ↓
"Suggest a catchy blog title about AI"
       ↓
ChatHuggingFace
       ↓
Qwen 2.5 1.5B
       ↓
Generated Blog Title

## Problems faced by langchain
When LangChain developers were building LLM applications, they initially faced a major problem: developers had to create and manage a separate chain for almost every individual step of an application.

For example, a simple RAG application might need:

User Question → Prompt → LLM → Parser → Retriever → Another Prompt → LLM → Final Answer

If every step had to be handled as a separate chain, several problems appeared:

1. Too much code – Developers had to write many separate chains even for simple workflows.
2. Difficult to connect chains – Passing the output of one chain into the next chain became cumbersome.
3. Poor reusability – A chain created for one workflow was often difficult to reuse in another workflow.
4. Hard to maintain – If one step changed, developers might need to modify multiple chains and their connections.
5. Complex debugging – Finding where an error occurred in a long sequence of individually connected chains was difficult.
6. Limited flexibility – Real applications often need branching, parallel execution, conditions, loops, and dynamic routing. Separate chains made these workflows harder to represent.
7. Data-flow problems – Managing inputs and outputs between multiple chains could become confusing, especially when several pieces of information had to be passed between steps.

## This led to LCEL
# LangChain introduced LCEL (LangChain Expression Language) to make it easier to compose multiple components into a single runnable pipeline.

Instead of manually creating a chain for every step:
Prompt Chain → LLM Chain → Parser Chain → ...

you can compose components:
chain = prompt | llm | parser

Here, | represents the flow of data from one component to the next.

So the key idea is:

Instead of creating a separate chain for every step, developers could compose reusable runnable components into larger chains.

Runnables are like unit of work.

Your give input to runnable, it processes it and gives output.

Every runnable follows commom interface.

As they follow common interface, they can be connected easily even in complex workflow.

The workflow created is itself a runnable.