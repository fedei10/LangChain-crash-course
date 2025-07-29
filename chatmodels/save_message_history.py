import os
from dotenv import load_dotenv
from supabase import create_client, Client
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from typing import List
from langchain_ollama import ChatOllama

# ------------------------------------------------------------------
# 0. load env & init Supabase
# ------------------------------------------------------------------
load_dotenv()
supabase: Client = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_KEY"]
)

# ------------------------------------------------------------------
# 1. Chat-history adapter for Supabase
# ------------------------------------------------------------------
class SupabaseChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str, supabase_client: Client):
        self.session_id = session_id
        self.supabase = supabase_client

    @property
    def messages(self) -> List[BaseMessage]:
        resp = (
            self.supabase.table("chat_history")
            .select("*")
            .eq("session_id", self.session_id)
            .order("timestamp", desc=False)
            .execute()
        )
        msgs = []
        for row in resp.data:
            if row["role"] == "user":
                msgs.append(HumanMessage(content=row["content"]))
            else:
                msgs.append(AIMessage(content=row["content"]))
        return msgs

    def add_user_message(self, message: str) -> None:
        self.supabase.table("chat_history").insert(
            {"session_id": self.session_id, "role": "user", "content": message}
        ).execute()

    def add_ai_message(self, message: str) -> None:
        self.supabase.table("chat_history").insert(
            {"session_id": self.session_id, "role": "assistant", "content": message}
        ).execute()

    def clear(self) -> None:
        (
            self.supabase.table("chat_history")
            .delete()
            .eq("session_id", self.session_id)
            .execute()
        )

# ------------------------------------------------------------------
# 2. start chat loop
# ------------------------------------------------------------------
SESSION_ID = "alice_123"
history = SupabaseChatMessageHistory(session_id=SESSION_ID, supabase_client=supabase)
model = ChatOllama(model="llama3.2", base_url="http://localhost:11434")

print("💬  Chat with history saved to Supabase. Type 'exit' to quit.")
while True:
    user = input("User: ")
    if user.strip().lower() == "exit":
        break

    history.add_user_message(user)
    ai_msg = model.invoke(history.messages)
    history.add_ai_message(ai_msg.content)
    print(f"AI:  {ai_msg.content}")