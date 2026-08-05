from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support assistant.'),
    MessagesPlaceholder(variable_name = 'chat_history'),
    ('human', '{query}')
])

chat_history = []
#load chat history from a file  
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())

print(chat_history)

#create prompt
prompt = chat_template.invoke({'chat_history': chat_history, 'query': 'where is my refund'})

print(prompt)