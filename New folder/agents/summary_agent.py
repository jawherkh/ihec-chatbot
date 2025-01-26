from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from typing import List, Dict
import sqlite3
import uuid

class SummaryAgent:
    def __init__(self, llm, db_path='conversation_history.db'):
        self.llm = llm
        self.db_path = db_path
        
        self.tools = [
            FunctionTool.from_defaults(
                fn=self.summarize_conversation,
                name="summarize_conversation",
                description="Summarizes the conversation history when it gets too long"
            )
        ]
        
        self.agent = ReActAgent.from_tools(
            tools=self.tools,
            llm=llm,
            verbose=True,
            system_prompt="You are a conversation summarizer. Maintain context between sessions using summaries."
        )
        
    def summarize_conversation(self, session_id: str, messages: List[Dict]) -> str:
        """Summarizes conversation history and stores it in SQLite"""
        context = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        summary = self.llm.complete(
            f"Summarize this conversation while preserving key details:\n{context}"
        ).text
        
        self.store_summary(session_id, summary)
        return summary
    
    def store_summary(self, session_id: str, summary: str):
        """Stores the summary in SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS summaries (
                session_id TEXT PRIMARY KEY,
                summary TEXT
            )
        ''')
        cursor.execute('''
            INSERT OR REPLACE INTO summaries (session_id, summary)
            VALUES (?, ?)
        ''', (session_id, summary))
        conn.commit()
        conn.close()
    
    def get_context(self, session_id: str) -> str:
        """Builds context from summary and recent messages"""
        summary = self.get_summary(session_id)
        history = self.get_history(session_id)
        
        context = []
        if summary:
            context.append(f"Previous Summary: {summary}")
            
        context.extend([f"{m['role']}: {m['content']}" for m in history[-3:]])
        return "\n".join(context)

    def get_summary(self, session_id: str) -> str:
        """Retrieves the summary from SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT summary FROM summaries WHERE session_id = ?', (session_id,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None

    def get_history(self, session_id: str) -> List[Dict]:
        """Retrieves conversation history from SQLite (implement as needed)"""
        # Placeholder for retrieving history; implement as per your database schema
        return []