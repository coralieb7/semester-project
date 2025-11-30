"""RAG System for Visual Guidelines - Simplified Implementation"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class VisualGuideline:
    """Visual guideline entry"""
    text: str
    source: str
    category: str = "general"


class VisualGuidelineRAG:
    """RAG system for visual guidelines"""
    
    def __init__(self,
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 collection_name: str = "visual_guidelines",
                 persist_directory: str = "./models/chroma_db"):
        """Initialize RAG system"""
        
        self.embedder = SentenceTransformer(embedding_model)
        
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except:
            self.collection = self.client.create_collection(name=collection_name)
    
    def add_guidelines(self, texts: List[str], categories: Optional[List[str]] = None):
        """Add guidelines to knowledge base"""
        if categories is None:
            categories = ["general"] * len(texts)
        
        embeddings = self.embedder.encode(texts)
        
        ids = [f"guideline_{i}" for i in range(len(texts))]
        metadatas = [{"category": cat} for cat in categories]
        
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
    
    def retrieve(self, query: str, top_k: int = 3, category: Optional[str] = None) -> List[VisualGuideline]:
        """Retrieve relevant guidelines"""
        
        query_embedding = self.embedder.encode(query)
        
        where_clause = {"category": category} if category else None
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where_clause
        )
        
        guidelines = []
        if results['documents']:
            for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                guidelines.append(VisualGuideline(
                    text=doc,
                    source="knowledge_base",
                    category=meta.get('category', 'general')
                ))
        
        return guidelines
    
    def enhance_prompt(self, base_prompt: str, top_k: int = 2) -> str:
        """Enhance prompt with guidelines"""
        
        guidelines = self.retrieve(base_prompt, top_k=top_k)
        
        if not guidelines:
            return base_prompt
        
        # Extract key terms
        guideline_terms = [g.text for g in guidelines if len(g.text.split()) < 15]
        
        if guideline_terms:
            enhanced = f"{base_prompt}, {', '.join(guideline_terms[:2])}"
            return enhanced
        
        return base_prompt
    
    def get_statistics(self):
        """Get knowledge base statistics"""
        return {
            "total_documents": self.collection.count(),
            "collection_name": self.collection.name
        }