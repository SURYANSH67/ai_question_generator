import os
import shutil
import pickle
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

class RetrieverManager:
    def __init__(self, api_key: Optional[str] = None, engine_type: str = "faiss", persist_dir: str = "retriever_db"):
        self.api_key = api_key
        self.engine_type = engine_type
        self.persist_dir = persist_dir
        self.db = None
        self.bm25_retriever = None
        
        if self.engine_type == "faiss" and api_key:
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=api_key,
                model="text-embedding-3-small"
            )
        else:
            self.embeddings = None

    def build_index(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Builds the selected retrieval index (FAISS or BM25) from chunks.
        """
        # Clear existing directory
        if os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir)
        os.makedirs(self.persist_dir, exist_ok=True)
            
        docs = [Document(
            page_content=chunk["text"],
            metadata={
                "chunk_id": chunk["chunk_id"],
                "page_number": chunk["page_number"],
                "word_count": chunk["word_count"]
            }
        ) for chunk in chunks]
        
        if self.engine_type == "faiss":
            if not self.embeddings:
                raise ValueError("OpenAI embeddings not initialized. Check API Key.")
            texts = [chunk["text"] for chunk in chunks]
            metadatas = [doc.metadata for doc in docs]
            self.db = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas
            )
            self.db.save_local(self.persist_dir)
        else:
            # BM25 local indexing
            self.bm25_retriever = BM25Retriever.from_documents(docs)
            # Save Document objects list instead of the complex Pydantic retriever object
            with open(os.path.join(self.persist_dir, "bm25_docs.pkl"), "wb") as f:
                pickle.dump(docs, f)
                
    def load_index(self) -> bool:
        """
        Loads the index from disk.
        """
        if not os.path.exists(self.persist_dir):
            return False
            
        if self.engine_type == "faiss":
            if not self.embeddings:
                return False
            if os.path.exists(os.path.join(self.persist_dir, "index.faiss")):
                try:
                    self.db = FAISS.load_local(
                        self.persist_dir, 
                        self.embeddings, 
                        allow_dangerous_deserialization=True
                    )
                    return True
                except Exception as e:
                    print(f"Error loading FAISS index: {e}")
                    return False
        else:
            # BM25 loader: loads Document objects and instantiates the retriever
            bm25_path = os.path.join(self.persist_dir, "bm25_docs.pkl")
            if os.path.exists(bm25_path):
                try:
                    with open(bm25_path, "rb") as f:
                        docs = pickle.load(f)
                    self.bm25_retriever = BM25Retriever.from_documents(docs)
                    return True
                except Exception as e:
                    print(f"Error recreating BM25 retriever: {e}")
                    return False
        return False
        
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves top_k relevant chunks for the given query.
        """
        if self.engine_type == "faiss":
            if not self.db:
                if not self.load_index():
                    raise ValueError("FAISS index is not initialized or loaded.")
            results = self.db.similarity_search_with_relevance_scores(query, k=top_k)
            parsed_results = []
            for doc, score in results:
                parsed_results.append({
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)
                })
            return parsed_results
        else:
            # BM25 retrieval
            if not self.bm25_retriever:
                if not self.load_index():
                    raise ValueError("BM25 index is not initialized or loaded.")
            
            try:
                # Set k for BM25 retriever
                self.bm25_retriever.k = top_k
                docs = self.bm25_retriever.invoke(query)
            except AttributeError as ae:
                # Force rebuild from session state if cached retriever is corrupt or missing standard attributes
                import streamlit as st
                if "pdf_chunks" in st.session_state and st.session_state.pdf_chunks:
                    self.build_index(st.session_state.pdf_chunks)
                    self.bm25_retriever.k = top_k
                    docs = self.bm25_retriever.invoke(query)
                else:
                    raise ae

            parsed_results = []
            for idx, doc in enumerate(docs):
                # Simulate a score based on rank since BM25 doesn't return raw normalized scores in standard interface
                simulated_score = 1.0 - (idx * 0.1)
                parsed_results.append({
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "score": max(0.1, simulated_score)
                })
            return parsed_results

    def search_by_topics(self, topics: List[str], questions_count: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves relevant contexts for a set of topics by performing similarity search for each.
        Consolidates results and eliminates duplicates.
        """
        all_chunks = {}
        chunks_per_topic = max(3, int((questions_count * 2) / len(topics))) if topics else 5
        
        for topic in topics:
            results = self.search(topic, top_k=chunks_per_topic)
            for chunk in results:
                chunk_id = chunk["metadata"].get("chunk_id")
                if chunk_id not in all_chunks or all_chunks[chunk_id]["score"] < chunk["score"]:
                    all_chunks[chunk_id] = {
                        "text": chunk["text"],
                        "metadata": chunk["metadata"],
                        "score": chunk["score"],
                        "query_topic": topic
                    }
                    
        # Sort by score and return
        sorted_results = sorted(all_chunks.values(), key=lambda x: x["score"], reverse=True)
        return sorted_results
