# rag_agent.py
import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
import json
import time
from typing import List, Dict, Any, Optional, Union

class ChillsRAGSystem:
    """RAG system for ChillsDB KG with ChromaDB embeddings"""
    
    def __init__(self, graph=None, collection_name: str = "chills_kg", use_llm: bool = False):
        self.graph = graph
        self.collection_name = collection_name
        self.use_llm = use_llm
        self.client = None
        self.collection = None
        self.llm_initialized = False
        
        print("🚀 Initializing Chills RAG System...")
        self._initialize_chroma()
        
        if self.graph is not None and self.collection.count() == 0:
            print("🆕 Building ChromaDB from KG...")
            self._build_chroma_from_kg()
        
        print(f"✅ RAG System ready with {self.collection.count()} documents")
    
    def _initialize_chroma(self):
        """Initialize ChromaDB with persistent storage"""
        try:
            self.client = chromadb.PersistentClient(path="./chroma_db")
            try:
                self.collection = self.client.get_collection(self.collection_name)
                print(f"✅ Loaded existing collection '{self.collection_name}'")
            except:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "ChillsDB Knowledge Graph"}
                )
                print(f"✅ Created new collection '{self.collection_name}'")
        except Exception as e:
            print(f"❌ ChromaDB initialization failed: {e}")
            raise
    
    def _build_chroma_from_kg(self):
        """Convert KG nodes to ChromaDB documents"""
        documents = []
        metadatas = []
        ids = []
        
        for node_id, data in self.graph.nodes(data=True):
            doc_text = (
                f"Media: {data.get('media', 'Unknown')}. "
                f"Polarity: {data.get('polarity', 'Neutral')}. "
                f"Rating: {data.get('rating', 0):.1f}. "
                f"Intensity: {data.get('intensity', 0):.1f}. "
                f"Valence: {data.get('valence', 0):.2f}. "
                f"Modality: {data.get('modality', 'Unknown')}."
            )
            documents.append(doc_text)
            metadatas.append({
                'media': data.get('media', 'Unknown'),
                'polarity': data.get('polarity', 'Neutral'),
                'rating': data.get('rating', 0),
                'intensity': data.get('intensity', 0),
                'valence': data.get('valence', 0),
                'modality': data.get('modality', 'Unknown'),
                'response': data.get('response', 'Unknown'),
                'context': data.get('context', 0)
            })
            ids.append(f"node_{node_id}")
        
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"✅ Added {len(documents)} nodes to ChromaDB")
    
    def query(self, question: str, n_results: int = 5) -> Union[List[Dict], str]:
        """Query ChromaDB and return results (matching your template's structure)"""
        print(f"🔍 Query: '{question}' (LLM: {'ON' if self.use_llm else 'OFF'})")
        
        try:
            results = self.collection.query(
                query_texts=[question],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            formatted_results = []
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "similarity_score": 1 / (1 + results['distances'][0][i]),
                    "node_id": results['ids'][0][i]
                })
            
            print(f"✅ Found {len(formatted_results)} relevant nodes")
            return formatted_results
            
        except Exception as e:
            print(f"❌ Query error: {e}")
            return []
    
    def get_statistics(self) -> Dict:
        """Return collection statistics"""
        return {
            "total_nodes": self.collection.count(),
            "collection_name": self.collection_name,
            "llm_available": self.llm_initialized
        }
    
    def debug_system(self):
        """Debug info"""
        print("\n" + "="*60)
        print("🐛 CHILLS RAG SYSTEM DEBUG REPORT")
        print("="*60)
        print(f"📊 Collection: {self.collection.count()} documents")
        print(f"🧠 LLM: {'Enabled' if self.use_llm else 'Disabled'}")
        print("="*60)