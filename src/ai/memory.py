import os
import chromadb
from chromadb.config import Settings
from typing import Optional, Dict, Any, List
from ..config import CHROMA_PERSIST_DIR, GEMINI_API_KEY

class MerchantMemory:
    """RAG-based Vector Memory for Learning and Auto-resolving Merchants & Friends."""
    
    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="merchant_memory",
            metadata={"hnsw:space": "cosine"}
        )
        self.tx_collection = self.client.get_or_create_collection(
            name="transaction_semantic_index",
            metadata={"hnsw:space": "cosine"}
        )

    def save_merchant_mapping(
        self,
        merchant_name: str,
        category: str,
        notes: str = "",
        upi_id: Optional[str] = None,
        is_permanent_rule: bool = True
    ):
        """Stores or updates a learned recipient category in vector memory."""
        clean_key = (merchant_name or "").strip().lower()
        if not clean_key:
            return

        rule_type = "Permanent Rule" if is_permanent_rule else "Variable Peer"
        doc_text = f"Recipient: {merchant_name}. UPI: {upi_id or 'N/A'}. Category: {category}. Rule: {rule_type}. Context: {notes}"
        metadata = {
            "merchant_name": merchant_name,
            "category": category,
            "notes": notes or "",
            "upi_id": upi_id or "",
            "is_permanent_rule": bool(is_permanent_rule)
        }

        self.collection.upsert(
            ids=[clean_key],
            documents=[doc_text],
            metadatas=[metadata]
        )

    def query_merchant(self, query_text: str, n_results: int = 1) -> Optional[Dict[str, Any]]:
        """Queries the vector database for matching merchant or friend memory."""
        if not query_text or self.collection.count() == 0:
            return None

        clean_query = query_text.strip().lower()
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )

        if results and results.get("ids") and len(results["ids"][0]) > 0:
            matched_id = results["ids"][0][0]
            distance = results["distances"][0][0] if "distances" in results and results["distances"] else 1.0
            
            # Match if ID matches or cosine distance is close
            if matched_id == clean_query or distance <= 0.65:
                meta = results["metadatas"][0][0]
                return {
                    "matched_merchant": meta.get("merchant_name"),
                    "category": meta.get("category"),
                    "notes": meta.get("notes"),
                    "upi_id": meta.get("upi_id"),
                    "is_permanent_rule": bool(meta.get("is_permanent_rule", True)),
                    "confidence": max(0.0, 1.0 - distance)
                }
        return None

    def index_transaction(self, tx_id: int, date: str, amount: float, recipient: str, category: str, notes: str):
        """Indexes a recorded transaction for semantic natural language Q&A."""
        doc_text = f"Transaction on {date}: ₹{amount} to {recipient} under {category}. Notes: {notes}"
        self.tx_collection.upsert(
            ids=[str(tx_id)],
            documents=[doc_text],
            metadatas={
                "tx_id": tx_id,
                "date": date,
                "amount": amount,
                "recipient": recipient,
                "category": category
            }
        )

    def search_transactions(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Performs semantic search across historical transactions."""
        if self.tx_collection.count() == 0:
            return []

        results = self.tx_collection.query(
            query_texts=[query],
            n_results=min(n_results, self.tx_collection.count())
        )

        matches = []
        if results and results.get("ids") and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                matches.append({
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i]
                })
        return matches

# Global Singleton Instance
memory_store = MerchantMemory()
