from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama



model = ChatOllama(model="llama3.2", base_url="http://localhost:11434")

chat_history = []


systel_message= SystemMessage(content="You are a pro gym coach")
chat_history.append(systel_message)


while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query))  # Add user message

    # Get AI response using history
    result = model.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response))  # Add AI message

    print(f"AI: {response}")


print("---- Message History ----")
print(chat_history)