# Chains 
Chains in LangChain are sequences of modular components—such as prompts, language models (LLMs), and parsers—linked together to automate multi-step tasks.  They function by passing the output of one component as input to the next, allowing developers to build complex, reusable AI workflows without manually managing the underlying data flow.

In modern LangChain, these chains are often constructed using the LangChain Expression Language (LCEL), which utilizes the pipe (|) operator to compose components like prompt | model | parser into streamlined pipelines. 

# Types of chains

# 1. Simple Sequential Chain
A Simple Chain in LangChain is the most basic workflow structure where the output of one component is directly passed as the input to the next, creating a linear pipeline with no branching or conditional logic.  

It typically follows the flow of Prompt Template → Large Language Model (LLM) → Output Parser. 

# 2. Sequential Chain
This is a more general form that supports multiple inputs and outputs per step.  It allows explicit mapping of variables, making it suitable for complex workflows where later steps depend on specific outputs from earlier steps.

# 3. Parallel Chain
Parallel chains in LangChain allow multiple independent sub-chains to execute concurrently, significantly reducing latency compared to sequential processing.

This is primarily implemented using RunnableParallel or by using a Python dictionary literal within a chain sequence, both of which distribute the same input to each branch and return a dictionary of results. 

# 3. Conditional Chain
Conditional chains in LangChain allow an application to dynamically route execution to different sub-chains based on specific conditions, such as the output of an intermediate step or the content of the input.  

This is primarily achieved using the RunnableBranch class (part of the LangChain Expression Language) or the ChoiceChain (from langchain-contrib), which act like if-else logic blocks within a pipeline.

Key Implementation Methods
RunnableBranch: The modern, idiomatic approach for routing. It evaluates a list of condition functions (lambdas) against the input; if a condition is True, the corresponding chain is executed. If no condition matches, a default chain is used