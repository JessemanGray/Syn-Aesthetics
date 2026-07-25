from data_processor import SynAestheticsDataProcessor
from rag_agent import ChillsRAGSystem

# Build processor and KG
processor = SynAestheticsDataProcessor()
df, G = processor.run_pipeline()

# Build RAG system from KG
rag = ChillsRAGSystem(graph=G, use_llm=False)

# Test query
results = rag.query("beneficial high intensity chills")
for r in results[:3]:
    print(f"Media: {r['metadata']['media']} — Score: {r['similarity_score']:.3f}")