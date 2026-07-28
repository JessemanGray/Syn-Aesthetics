# chills_rag.py
import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
import json
import time
from typing import List, Dict, Any, Optional, Union

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠️ Ollama not installed. Install with: pip install ollama")

class ChillsRAGSystem:
    """RAG system for ChillsDB KG with Ollama SLM as aesthetic alignment layer"""
    
    def __init__(self, graph=None, collection_name: str = "chills_kg", use_llm: bool = False, model_name: str = "gemma2:2b"):
        self.graph = graph
        self.collection_name = collection_name
        self.use_llm = use_llm
        self.model_name = model_name
        self.client = None
        self.collection = None
        self.llm_initialized = False
        
        print("🚀 Initializing Chills RAG System...")
        self._initialize_chroma()
        
        if self.graph is not None and self.collection.count() == 0:
            print("🆕 Building ChromaDB from KG...")
            self._build_chroma_from_kg()
        
        if self.use_llm and OLLAMA_AVAILABLE:
            try:
                print(f"🧠 Loading {self.model_name} via Ollama...")
                ollama.chat(model=self.model_name, messages=[{"role": "user", "content": "test"}])
                self.llm_initialized = True
                print(f"✅ {self.model_name} loaded successfully")
            except Exception as e:
                print(f"⚠️ {self.model_name} not available: {e}")
                print("   Install with: ollama pull gemma2:2b")
                self.use_llm = False
        elif self.use_llm and not OLLAMA_AVAILABLE:
            print("⚠️ Ollama not available. Install with: pip install ollama")
            self.use_llm = False
        
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
        """Convert KG nodes to ChromaDB documents with blob metadata"""
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
                f"Modality: {data.get('modality', 'Unknown')}. "
                f"Response: {data.get('response', 'Unknown')}. "
                f"Context: {data.get('context', 0)}/4. "
                f"Chills Ratio: {data.get('chills_ratio', 0):.2f}. "
                f"Mean Intensity: {data.get('mean_intensity', 0):.1f}. "
                f"Mean Valence: {data.get('mean_valence', 0):.2f}. "
                f"Mean Arousal: {data.get('mean_arousal', 0):.2f}. "
                f"Mean Liking: {data.get('mean_liking', 0):.1f}. "
                f"Blob Polarity: {data.get('blob_polarity', 'neutral')}."
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
                'context': data.get('context', 0),
                'chills_ratio': data.get('chills_ratio', 0),
                'mean_intensity': data.get('mean_intensity', 0),
                'mean_valence': data.get('mean_valence', 0),
                'mean_arousal': data.get('mean_arousal', 0),
                'mean_liking': data.get('mean_liking', 0),
                'blob_polarity': data.get('blob_polarity', 'neutral')
            })
            ids.append(f"node_{node_id}")
        
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"✅ Added {len(documents)} enriched nodes to ChromaDB")
    
    def query(self, question: str, n_results: int = 5, generate: bool = None) -> Union[List[Dict], str]:
        """Query ChromaDB and return results or generated response."""
        use_generation = generate if generate is not None else self.use_llm
        print(f"🔍 Query: '{question}' (LLM: {'ON' if use_generation else 'OFF'})")
        
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
            
            if use_generation and self.llm_initialized and formatted_results:
                try:
                    context = "\n\n".join([r['content'] for r in formatted_results[:3]])
                    prompt = f"""You are an aesthetic alignment layer for ChillsDB. Use the following context to answer the user's question. Reference polarity, intensity, chills ratio, and valence where relevant.

Context:
{context}

Question: {question}

Answer concisely and align with aesthetic criteria."""
                    
                    response = ollama.chat(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    return response['message']['content']
                except Exception as e:
                    print(f"⚠️ Generation failed: {e}")
                    return formatted_results
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ Query error: {e}")
            return []
    
    def get_statistics(self) -> Dict:
        return {
            "total_nodes": self.collection.count(),
            "collection_name": self.collection_name,
            "llm_available": self.llm_initialized,
            "llm_model": self.model_name if self.llm_initialized else "None"
        }
    
    def debug_system(self):
        print("\n" + "="*60)
        print("🐛 CHILLS RAG SYSTEM DEBUG REPORT")
        print("="*60)
        print(f"📊 Collection: {self.collection.count()} documents")
        print(f"🧠 LLM: {'Enabled' if self.use_llm else 'Disabled'}")
        print(f"🤖 Model: {self.model_name if self.llm_initialized else 'Not available'}")
        print("="*60)