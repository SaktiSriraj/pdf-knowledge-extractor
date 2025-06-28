import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from langchain.schema import Document
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME, TOP_K, LANGCHAIN_CHUNK_SIZE, LANGCHAIN_MAX_TOKENS
from model_loader import generate_response
from indexer import langchain_indexer

class EnhancedLangChainRAG:
    """Advanced RAG implementation using LangChain components"""
    
    def __init__(self):
        self.embedding_model = None
        self.retriever = None
        self.qa_chain = None
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=5  # Remember last 5 exchanges
        )
        
    def initialize_embedding_model(self):
        """Initialize embedding model for retrieval"""
        try:
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            return self.embedding_model
        except Exception as e:
            raise Exception(f"Failed to load embedding model: {str(e)}")
    
    def create_advanced_retriever(self, vector_store, retrieval_strategy: str = "similarity"):
        """Create advanced retriever with different strategies"""
        try:
            if retrieval_strategy == "similarity":
                # Standard similarity search
                retriever = vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": TOP_K}
                )
            
            elif retrieval_strategy == "mmr":
                # Maximum Marginal Relevance for diversity
                retriever = vector_store.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": TOP_K,
                        "fetch_k": TOP_K * 2,  # Fetch more candidates
                        "lambda_mult": 0.7     # Balance relevance vs diversity
                    }
                )
            
            elif retrieval_strategy == "similarity_score":
                # Score-based threshold
                retriever = vector_store.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={
                        "k": TOP_K,
                        "score_threshold": 0.5
                    }
                )
            
            else:
                # Default to similarity
                retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})
            
            self.retriever = retriever
            return retriever
            
        except Exception as e:
            raise Exception(f"Failed to create advanced retriever: {str(e)}")
    
    def create_compression_retriever(self, base_retriever):
        """Create contextual compression retriever for better relevance"""
        try:
            # Simple compression based on query keywords
            class SimpleCompressor:
                def compress_documents(self, documents, query):
                    # Simple compression based on query keywords
                    query_words = set(query.lower().split())
                    
                    compressed_docs = []
                    for doc in documents:
                        doc_words = set(doc.page_content.lower().split())
                        overlap = len(query_words.intersection(doc_words))
                        
                        if overlap > 0:
                            compressed_docs.append(doc)
                    
                    return compressed_docs[:TOP_K]
            
            compressor = SimpleCompressor()
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever
            )
            
            return compression_retriever
            
        except Exception as e:
            print(f"Could not create compression retriever: {str(e)}")
            return base_retriever
    
    def create_qa_prompts(self, query_type: str = "general") -> PromptTemplate:
        """Create sophisticated prompts for different query types"""
        
        base_context = """You are an expert document analyst with deep expertise in information extraction and synthesis. Your task is to provide comprehensive, accurate, and insightful responses based on the provided document context.

CORE PRINCIPLES:
1. ACCURACY: Base all responses strictly on the provided context
2. COMPLETENESS: Address all aspects of the question thoroughly
3. CLARITY: Use clear, professional language with proper structure
4. EVIDENCE: Cite specific information from the documents
5. INSIGHT: Provide analytical depth beyond surface-level information

CONTEXT DOCUMENTS:
{context}

QUESTION: {question}

RESPONSE GUIDELINES:"""
        
        if query_type == "summary":
            specific_guidelines = """
- Provide a comprehensive summary covering all major themes
- Include key details, statistics, and findings
- Organize information hierarchically (main points → supporting details)
- Highlight relationships between different concepts
- Conclude with the most significant insights or implications"""
            
        elif query_type == "analysis":
            specific_guidelines = """
- Conduct thorough analysis of the information presented
- Examine underlying patterns, trends, or relationships
- Evaluate strengths, weaknesses, opportunities, or threats as relevant
- Provide critical assessment of arguments or evidence
- Draw meaningful conclusions based on the analysis"""
            
        elif query_type == "comparison":
            specific_guidelines = """
- Systematically compare and contrast the relevant elements
- Create clear distinctions between similarities and differences
- Provide specific examples for each point of comparison
- Evaluate the significance of identified differences
- Conclude with insights about the comparative analysis"""
            
        elif query_type == "explanation":
            specific_guidelines = """
- Provide detailed, step-by-step explanations
- Break down complex concepts into understandable components
- Use examples and illustrations from the documents
- Explain the reasoning or logic behind processes or decisions
- Address potential questions or clarifications"""
            
        else:  # general
            specific_guidelines = """
- Provide a comprehensive response addressing all aspects of the question
- Structure your answer with clear organization and flow
- Include relevant details, examples, and supporting evidence
- Offer additional context that enhances understanding
- Ensure completeness while maintaining focus on the core question"""
        
        full_prompt = base_context + specific_guidelines + """

Please provide your detailed response:"""
        
        return PromptTemplate(
            template=full_prompt,
            input_variables=["context", "question"]
        )
    
    def analyze_query_intent(self, query: str) -> str:
        """Analyze query to determine the best response approach"""
        query_lower = query.lower()
        
        # Define intent patterns
        intent_patterns = {
            'summary': ['summarize', 'summary', 'overview', 'main points', 'key points'],
            'analysis': ['analyze', 'analysis', 'evaluate', 'assessment', 'examine'],
            'comparison': ['compare', 'contrast', 'difference', 'versus', 'vs', 'between'],
            'explanation': ['explain', 'how', 'why', 'what is', 'define', 'describe'],
            'list': ['list', 'enumerate', 'identify', 'what are'],
            'specific': ['when', 'where', 'who', 'which', 'specific']
        }
        
        # Count matches for each intent
        intent_scores = {}
        for intent, patterns in intent_patterns.items():
            score = sum(1 for pattern in patterns if pattern in query_lower)
            if score > 0:
                intent_scores[intent] = score
        
        # Return the intent with highest score, default to 'general'
        if intent_scores:
            return max(intent_scores, key=intent_scores.get)
        else:
            return 'general'

# Global RAG instance
enhanced_rag = EnhancedLangChainRAG()

def initialize_embedding_model():
    """Initialize embedding model using enhanced RAG"""
    return enhanced_rag.initialize_embedding_model()

def retrieve_chunks_advanced(query: str, index, chunks: List[str], metadata: Optional[List[Dict]] = None, top_k: int = TOP_K) -> Tuple[List[str], List[Dict]]:
    """Advanced chunk retrieval using LangChain when available"""
    try:
        # Initialize embedding model if needed
        if enhanced_rag.embedding_model is None:
            enhanced_rag.initialize_embedding_model()
        
        # Try LangChain vector store first
        try:
            if langchain_indexer.vector_store is not None:
                # Use LangChain similarity search
                docs = langchain_indexer.vector_store.similarity_search_with_score(query, k=top_k)
                
                retrieved_chunks = []
                chunk_info = []
                
                for i, (doc, score) in enumerate(docs):
                    retrieved_chunks.append(doc.page_content)
                    
                    info = {
                        "index": doc.metadata.get('chunk_index', i),
                        "score": float(score),
                        "preview": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content,
                        "langchain_enhanced": True
                    }
                    
                    # Add metadata from document
                    info.update(doc.metadata)
                    chunk_info.append(info)
                
                return retrieved_chunks, chunk_info
        except Exception as e:
            print(f"LangChain retrieval failed, using fallback: {str(e)}")
        
        # Fallback to traditional method
        embedding_model = enhanced_rag.embedding_model
        query_embedding = embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        
        scores, indices = index.search(query_embedding, top_k)
        
        retrieved_chunks = []
        chunk_info = []
        
        for i, idx in enumerate(indices[0]):
            if idx < len(chunks):
                chunk = chunks[idx]
                score = float(scores[0][i])
                
                retrieved_chunks.append(chunk)
                
                info = {
                    "index": int(idx),
                    "score": score,
                    "preview": chunk[:150] + "..." if len(chunk) > 150 else chunk,
                    "langchain_enhanced": False
                }
                
                if metadata and idx < len(metadata):
                    info.update(metadata[idx])
                
                chunk_info.append(info)
        
        return retrieved_chunks, chunk_info
        
    except Exception as e:
        raise Exception(f"Failed to retrieve chunks: {str(e)}")

def run_rag_pipeline(query: str, index, chunks: List[str], metadata: Optional[List[Dict]] = None, max_tokens: int = LANGCHAIN_MAX_TOKENS) -> Tuple[str, List[Dict]]:
    """Run enhanced RAG pipeline with LangChain components"""
    try:
        if not query.strip():
            raise ValueError("Query cannot be empty")
        
        # Analyze query intent for better prompting
        query_intent = enhanced_rag.analyze_query_intent(query)
        
        # Retrieve relevant chunks using advanced methods
        enhanced_top_k = min(8, len(chunks))  # Get more chunks for better context
        retrieved_chunks, chunk_info = retrieve_chunks_advanced(
            query, index, chunks, metadata, enhanced_top_k
        )
        
        if not retrieved_chunks:
            return "No relevant information found in the documents.", []
        
        # Create enhanced context with LangChain-style organization
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks):
            source_info = ""
            if chunk_info and i < len(chunk_info):
                source = chunk_info[i].get('source_file', chunk_info[i].get('source', 'Unknown'))
                score = chunk_info[i].get('score', 0)
                source_info = f" [Source: {source}, Relevance: {score:.3f}]"
            
            context_parts.append(f"=== DOCUMENT SECTION {i+1}{source_info} ===\n{chunk}\n")
        
        context = "\n".join(context_parts)
        
        # Smart context length management
        max_context_length = 6000
        if len(context) > max_context_length:
            # Prioritize highest scoring chunks
            sorted_chunks_info = sorted(
                zip(retrieved_chunks, chunk_info), 
                key=lambda x: x[1].get('score', 0)
            )
            
            context_parts = []
            current_length = 0
            
            for i, (chunk, info) in enumerate(sorted_chunks_info):
                source = info.get('source_file', info.get('source', 'Unknown'))
                score = info.get('score', 0)
                section = f"=== DOCUMENT SECTION {i+1} [Source: {source}, Relevance: {score:.3f}] ===\n{chunk}\n"
                
                if current_length + len(section) <= max_context_length:
                    context_parts.append(section)
                    current_length += len(section)
                else:
                    break
            
            context = "\n".join(context_parts)
            if len(context_parts) < len(retrieved_chunks):
                context += f"\n\n[Note: {len(retrieved_chunks) - len(context_parts)} additional relevant sections were found but truncated for optimal processing]"
        
        # Create intent-specific prompt using LangChain
        prompt_template = enhanced_rag.create_qa_prompts(query_intent)
        
        # Format the prompt
        formatted_prompt = prompt_template.format(
            context=context,
            question=query
        )
        
        # Generate response with enhanced settings
        response = generate_response(
            formatted_prompt,
            max_tokens=max_tokens,
            temperature=0.7
        )
        
        # Add LangChain processing info to chunk info
        for info in chunk_info:
            info.update({
                "query_intent": query_intent,
                "prompt_strategy": "langchain_enhanced",
                "context_length": len(context),
                "processing_method": "Enhanced LangChain RAG"
            })
        
        return response, chunk_info
        
    except Exception as e:
        raise Exception(f"Enhanced RAG pipeline failed: {str(e)}")

def get_rag_capabilities() -> Dict[str, Any]:
    """Get information about enhanced RAG capabilities"""
    return {
        "langchain_features": [
            "Advanced Vector Store Integration",
            "Multiple Retrieval Strategies (Similarity, MMR, Score-based)",
            "Intent-based Query Analysis",
            "Enhanced Prompt Engineering",
            "Document Ranking and Scoring"
        ],
        "retrieval_strategies": [
            "Similarity Search",
            "Maximum Marginal Relevance (MMR)",
            "Score-based Threshold Filtering"
        ],
        "prompt_engineering": [
            "Intent-specific Prompting",
            "Query Type Detection",
            "Context-aware Formatting",
            "Evidence-based Responses"
        ],
        "advanced_capabilities": [
            "Multi-document Synthesis",
            "Hierarchical Information Organization",
            "Relevance Score Integration",
            "Source Attribution",
            "Context Length Optimization"
        ]
    }

def get_simple_answer(query: str, index, chunks: List[str], metadata: Optional[List[Dict]] = None) -> str:
    """Get a comprehensive answer using enhanced RAG"""
    try:
        response, _ = run_rag_pipeline(query, index, chunks, metadata, max_tokens=1500)
        return response
    except Exception as e:
        return f"Enhanced RAG Error: {str(e)}"