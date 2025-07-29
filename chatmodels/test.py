# 1_chat_model_basic.py

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

# Create a prompt
template = """Question: {question}

Answer: Let's think step by step."""
prompt = ChatPromptTemplate.from_template(template)

# Load your local LLM
model = OllamaLLM(model="llama3.2",base_url="http://localhost:11434")  # Or try llama3 if it's the tag

# Compose chain
chain = prompt | model

# Call the model
response = chain.invoke({"question": "What is MoE in AI?"})
print(response)
