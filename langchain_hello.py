"""Day 1: the smallest possible LCEL chain."""
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# init_chat_model infers the provider from the model string
model = init_chat_model("claude-sonnet-4-6", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an SQF food-safety expert. Be concise. Use plain hyphens, not em dashes."),
    ("human", "{question}"),
])

# The chain: prompt -> model -> string parser
chain = prompt | model | StrOutputParser()

# Run it
answer = chain.invoke({"question": "In two sentences, what is the role of the SQF practitioner?"})
print(answer)