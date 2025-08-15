"""
===============================================================================
HAYSTACK COMPLETE BEGINNER TUTORIAL
===============================================================================

This tutorial teaches you how to build a complete RAG (Retrieval-Augmented 
Generation) system using Haystack. You'll learn to:

1. Set up Haystack components
2. Create and manage document stores
3. Build search and question-answering pipelines
4. Process different types of documents

PREREQUISITES:
- Python 3.8 or higher
- pip install haystack-ai python-dotenv
- OpenAI API key (get from https://platform.openai.com/)

SETUP:
Create a .env file with:
OPENAI_API_KEY=your_api_key_here
===============================================================================
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Haystack imports - these are the building blocks we'll use
from haystack import Pipeline, Document
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders import PromptBuilder
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.converters import TextFileToDocument
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter

# Set up logging so we can see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HaystackTutorial:
    """
    A complete tutorial class that demonstrates all major Haystack concepts.
    
    This class will teach you:
    - How to set up components
    - How to create pipelines 
    - How to build a RAG system
    - How to search and generate answers
    """
    
    def __init__(self):
        """Initialize the tutorial by setting up environment and basic components."""
        self.setup_environment()
        self.document_store = None
        self.indexing_pipeline = None
        self.search_pipeline = None
        self.rag_pipeline = None
        
    def setup_environment(self):
        """
        STEP 1: Environment Setup
        
        This loads your API key and validates the setup.
        """
        print("🚀 STEP 1: Setting up environment...")
        
        # Load environment variables from .env file
        load_dotenv()
        
        # Check if OpenAI API key exists
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "❌ No OPENAI_API_KEY found in .env file.\n"
                "Please create a .env file with:\n"
                "OPENAI_API_KEY=your_actual_api_key_here"
            )
        
        print("✅ Environment setup complete!")
        print(f"✅ API key found: {self.api_key[:10]}...")
        
    def create_sample_documents(self):
        """
        STEP 2: Create Sample Documents
        
        This creates realistic sample documents that we'll use to demonstrate
        Haystack's capabilities. In real life, these would be your actual documents.
        """
        print("\n📚 STEP 2: Creating sample documents...")
        
        # Create data directory
        data_dir = Path("tutorial_data")
        data_dir.mkdir(exist_ok=True)
        
        # Sample documents with realistic content
        documents_content = {
            "company_handbook.txt": """
            EMPLOYEE HANDBOOK - TECHCORP SOLUTIONS
            =====================================
            
            VACATION POLICY:
            All full-time employees are entitled to 15 days of paid vacation per year.
            Vacation time must be requested at least 2 weeks in advance.
            Unused vacation days can be carried over to the next year, up to 5 days maximum.
            
            REMOTE WORK POLICY:
            Employees may work remotely up to 3 days per week with manager approval.
            Remote work equipment will be provided by the company.
            All remote workers must be available during core business hours (9 AM - 3 PM EST).
            
            HEALTH BENEFITS:
            The company covers 80% of health insurance premiums for employees.
            Dental and vision coverage is available with employee contribution.
            Annual health check-ups are fully covered.
            """,
            
            "technical_documentation.txt": """
            TECHCORP API DOCUMENTATION
            =========================
            
            AUTHENTICATION:
            All API requests must include an Authorization header with a valid Bearer token.
            Tokens expire after 24 hours and must be refreshed.
            
            RATE LIMITING:
            Free tier: 100 requests per hour
            Pro tier: 1000 requests per hour
            Enterprise tier: Unlimited requests
            
            ERROR CODES:
            200 - Success
            401 - Unauthorized (invalid or expired token)
            429 - Rate limit exceeded
            500 - Internal server error
            
            ENDPOINTS:
            GET /api/users - Retrieve user information
            POST /api/data - Submit new data
            PUT /api/data/{id} - Update existing data
            DELETE /api/data/{id} - Delete data
            """,
            
            "project_guidelines.txt": """
            PROJECT MANAGEMENT GUIDELINES
            ============================
            
            SPRINT PLANNING:
            Sprints are 2 weeks long and begin every other Monday.
            Sprint planning meetings occur on the Friday before each sprint.
            All team members must attend sprint planning meetings.
            
            CODE REVIEW PROCESS:
            All code must be reviewed by at least one senior developer.
            Pull requests should be submitted at least 24 hours before deployment.
            Code reviews should focus on functionality, security, and maintainability.
            
            TESTING REQUIREMENTS:
            Unit tests must cover at least 80% of the codebase.
            Integration tests are required for all API endpoints.
            Manual testing must be performed before production deployment.
            """
        }
        
        # Write documents to files
        for filename, content in documents_content.items():
            file_path = data_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Created: {filename}")
        
        print(f"✅ All sample documents created in {data_dir}")
        return data_dir
    
    def setup_document_store(self):
        """
        STEP 3: Document Store Setup
        
        A Document Store is like a smart database that stores your documents
        in a way that makes them searchable. We're using InMemoryDocumentStore
        for simplicity, but Haystack supports many databases.
        """
        print("\n🗃️ STEP 3: Setting up document store...")
        
        # Create an in-memory document store
        # This stores documents in RAM (fast but not persistent)
        self.document_store = InMemoryDocumentStore()
        
        print("✅ Document store created!")
        print("ℹ️  Using InMemoryDocumentStore (data will be lost when program ends)")
        print("ℹ️  For production, consider Elasticsearch, Weaviate, or other persistent stores")
        
    def create_indexing_pipeline(self):
        """
        STEP 4: Create Indexing Pipeline
        
        An indexing pipeline processes your documents and makes them searchable.
        It's like creating a super-detailed index for a book, but much smarter.
        
        Our pipeline does:
        1. Convert files to Documents
        2. Split large documents into smaller chunks
        3. Create embeddings (numerical representations)
        4. Store everything in the document store
        """
        print("\n🔄 STEP 4: Creating indexing pipeline...")
        
        # Component 1: Convert text files to Document objects
        file_converter = TextFileToDocument()
        
        # Component 2: Split documents into smaller chunks
        # This helps with better search results and fits LLM context limits
        document_splitter = DocumentSplitter(
            split_by="sentence",  # Split by sentences
            split_length=3,       # 3 sentences per chunk
            split_overlap=1       # 1 sentence overlap between chunks
        )
        
        # Component 3: Create embeddings using OpenAI
        # Embeddings convert text to numbers that represent meaning
        document_embedder = OpenAIDocumentEmbedder(
            model="text-embedding-3-small"  # Efficient embedding model
        )
        
        # Component 4: Write documents to the store
        document_writer = DocumentWriter(document_store=self.document_store)
        
        # Create the pipeline and connect components
        self.indexing_pipeline = Pipeline()
        
        # Add components to pipeline
        self.indexing_pipeline.add_component("file_converter", file_converter)
        self.indexing_pipeline.add_component("splitter", document_splitter)
        self.indexing_pipeline.add_component("embedder", document_embedder)
        self.indexing_pipeline.add_component("writer", document_writer)
        
        # Connect components (data flows from one to the next)
        self.indexing_pipeline.connect("file_converter", "splitter")
        self.indexing_pipeline.connect("splitter", "embedder")
        self.indexing_pipeline.connect("embedder", "writer")
        
        print("✅ Indexing pipeline created!")
        print("📋 Pipeline components:")
        print("   1. TextFileToDocument - Converts files to Document objects")
        print("   2. DocumentSplitter - Splits large documents into chunks") 
        print("   3. OpenAIDocumentEmbedder - Creates embeddings for semantic search")
        print("   4. DocumentWriter - Stores documents in document store")
        
    def index_documents(self, data_dir):
        """
        STEP 5: Index Documents
        
        This runs our indexing pipeline on the sample documents.
        After this step, documents will be searchable.
        """
        print(f"\n📥 STEP 5: Indexing documents from {data_dir}...")
        
        # Get all text files from the data directory
        file_paths = list(data_dir.glob("*.txt"))
        
        if not file_paths:
            raise ValueError(f"No .txt files found in {data_dir}")
        
        print(f"Found {len(file_paths)} files to index:")
        for file_path in file_paths:
            print(f"  - {file_path.name}")
        
        # Run the indexing pipeline
        print("\n🔄 Running indexing pipeline...")
        print("⏳ This may take a moment (creating embeddings with OpenAI)...")
        
        try:
            result = self.indexing_pipeline.run({
                "file_converter": {"sources": file_paths}
            })
            
            # Check how many documents were processed
            documents_written = result["writer"]["documents_written"]
            
            print(f"✅ Successfully indexed {documents_written} document chunks!")
            print(f"📊 Document store now contains {self.document_store.count_documents()} documents")
            
        except Exception as e:
            print(f"❌ Error during indexing: {e}")
            raise
    
    def create_search_pipeline(self):
        """
        STEP 6: Create Search Pipeline
        
        A search pipeline finds relevant documents based on a query.
        This is the "Retrieval" part of RAG.
        """
        print("\n🔍 STEP 6: Creating search pipeline...")
        
        # Component 1: Convert query text to embeddings
        query_embedder = OpenAITextEmbedder(
            model="text-embedding-3-small"
        )
        
        # Component 2: Search for similar documents
        retriever = InMemoryEmbeddingRetriever(
            document_store=self.document_store,
            top_k=3  # Return top 3 most relevant documents
        )
        
        # Create and configure the search pipeline
        self.search_pipeline = Pipeline()
        self.search_pipeline.add_component("query_embedder", query_embedder)
        self.search_pipeline.add_component("retriever", retriever)
        
        # Connect the components
        self.search_pipeline.connect("query_embedder.embedding", "retriever.query_embedding")
        
        print("✅ Search pipeline created!")
        print("📋 Pipeline components:")
        print("   1. OpenAITextEmbedder - Converts query to embedding")
        print("   2. InMemoryEmbeddingRetriever - Finds similar documents")
    
    def create_rag_pipeline(self):
        """
        STEP 7: Create RAG Pipeline
        
        RAG = Retrieval-Augmented Generation
        This pipeline combines search (retrieval) with AI text generation
        to answer questions based on your documents.
        """
        print("\n🤖 STEP 7: Creating RAG pipeline...")
        
        # Template for prompting the AI
        # This tells the AI how to use the retrieved documents
        prompt_template = """
        Answer the question based on the given context.
        
        Context:
        {% for document in documents %}
        {{ document.content }}
        {% endfor %}
        
        Question: {{ question }}
        
        Answer: Provide a clear, accurate answer based only on the information in the context above. If the answer is not in the context, say "I don't have enough information to answer this question."
        """
        
        # Component 1: Convert query to embeddings (same as search)
        query_embedder = OpenAITextEmbedder(
            model="text-embedding-3-small"
        )
        
        # Component 2: Retrieve relevant documents
        retriever = InMemoryEmbeddingRetriever(
            document_store=self.document_store,
            top_k=3
        )
        
        # Component 3: Build the prompt with context and question
        prompt_builder = PromptBuilder(template=prompt_template)
        
        # Component 4: Generate answer using OpenAI
        generator = OpenAIGenerator(
            model="gpt-3.5-turbo",
            generation_kwargs={"temperature": 0.1}  # Low temperature for consistent answers
        )
        
        # Create and configure the RAG pipeline
        self.rag_pipeline = Pipeline()
        self.rag_pipeline.add_component("query_embedder", query_embedder)
        self.rag_pipeline.add_component("retriever", retriever)
        self.rag_pipeline.add_component("prompt_builder", prompt_builder)
        self.rag_pipeline.add_component("generator", generator)
        
        # Connect all components
        self.rag_pipeline.connect("query_embedder.embedding", "retriever.query_embedding")
        self.rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
        self.rag_pipeline.connect("prompt_builder.prompt", "generator.prompt")
        
        print("✅ RAG pipeline created!")
        print("📋 Pipeline components:")
        print("   1. OpenAITextEmbedder - Converts question to embedding")
        print("   2. InMemoryEmbeddingRetriever - Finds relevant documents")
        print("   3. PromptBuilder - Combines documents and question into prompt")
        print("   4. OpenAIGenerator - Generates answer using GPT")
        
    def demonstrate_search(self):
        """
        STEP 8: Demonstrate Search
        
        Shows how to search for documents without generating answers.
        This is useful when you just want to find relevant information.
        """
        print("\n🔍 STEP 8: Demonstrating document search...")
        
        # Example search queries
        search_queries = [
            "vacation policy",
            "API authentication",
            "code review process"
        ]
        
        for query in search_queries:
            print(f"\n🔎 Searching for: '{query}'")
            
            # Run the search pipeline
            result = self.search_pipeline.run({"query_embedder": {"text": query}})
            
            # Get the retrieved documents
            documents = result["retriever"]["documents"]
            
            print(f"📄 Found {len(documents)} relevant documents:")
            
            for i, doc in enumerate(documents, 1):
                # Show a snippet of each document
                content_preview = doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
                print(f"\n   Document {i}:")
                print(f"   Score: {doc.score:.3f}")
                print(f"   Content: {content_preview}")
    
    def demonstrate_rag(self):
        """
        STEP 9: Demonstrate RAG Q&A
        
        Shows how to ask questions and get AI-generated answers
        based on your documents.
        """
        print("\n🤖 STEP 9: Demonstrating RAG question answering...")
        
        # Example questions
        questions = [
            "How many vacation days do employees get?",
            "What is the rate limit for the free tier API?", 
            "How long are sprints and when do they start?",
            "What percentage of health insurance does the company cover?",
            "What are the testing requirements for code?"
        ]
        
        for question in questions:
            print(f"\n❓ Question: {question}")
            print("⏳ Generating answer...")
            
            # Run the RAG pipeline
            result = self.rag_pipeline.run({
                "query_embedder": {"text": question},
                "prompt_builder": {"question": question}
            })
            
            # Get the generated answer
            answer = result["generator"]["replies"][0]
            
            print(f"🎯 Answer: {answer}")
            
            # Show which documents were used
            # Debug: print available keys to troubleshoot
            if "retriever" not in result:
                print(f"Available keys in result: {list(result.keys())}")
                documents = []
            else:
                documents = result["retriever"]["documents"]
            print(f"📚 Based on {len(documents)} relevant documents")
    
    def interactive_mode(self):
        """
        STEP 10: Interactive Mode
        
        Allows you to ask your own questions interactively.
        This is the fun part where you can explore your data!
        """
        print("\n💬 STEP 10: Interactive Q&A Mode")
        print("=" * 50)
        print("Now you can ask your own questions!")
        print("Type 'quit' to exit")
        print("=" * 50)
        
        while True:
            # Get user question
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Thanks for using the Haystack tutorial!")
                break
            
            if not question:
                continue
            
            try:
                print("⏳ Searching and generating answer...")
                
                # Run RAG pipeline
                result = self.rag_pipeline.run({
                    "query_embedder": {"text": question},
                    "prompt_builder": {"question": question}
                })
                
                # Display answer
                answer = result["generator"]["replies"][0]
                documents = result.get("retriever", {}).get("documents", [])
                
                print(f"\n🎯 Answer: {answer}")
                print(f"📚 (Based on {len(documents)} relevant documents)")
                
                # Optionally show source documents
                show_sources = input("\n🔍 Show source documents? (y/n): ").lower()
                if show_sources in ['y', 'yes']:
                    for i, doc in enumerate(documents, 1):
                        print(f"\n📄 Document {i} (Score: {doc.score:.3f}):")
                        print(f"   {doc.content[:300]}...")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                print("Please try a different question.")
    
    def run_complete_tutorial(self):
        """
        Main method that runs the complete tutorial from start to finish.
        """
        print("=" * 80)
        print("🎓 HAYSTACK COMPLETE TUTORIAL")
        print("=" * 80)
        print("This tutorial will teach you everything about building RAG systems with Haystack!")
        print()
        
        try:
            # Run all tutorial steps
            data_dir = self.create_sample_documents()
            self.setup_document_store()
            self.create_indexing_pipeline()
            self.index_documents(data_dir)
            self.create_search_pipeline()
            self.create_rag_pipeline()
            self.demonstrate_search()
            self.demonstrate_rag()
            
            # Optional interactive mode
            print("\n" + "=" * 50)
            interactive = input("🎮 Would you like to try interactive mode? (y/n): ").lower()
            if interactive in ['y', 'yes']:
                self.interactive_mode()
            
            print("\n🎉 Tutorial completed successfully!")
            print("\n📚 What you learned:")
            print("   ✅ How to set up Haystack components")
            print("   ✅ How to create and use document stores")
            print("   ✅ How to build indexing pipelines")
            print("   ✅ How to create search pipelines")
            print("   ✅ How to build RAG pipelines")
            print("   ✅ How to search documents")
            print("   ✅ How to generate AI answers from your data")
            
        except Exception as e:
            print(f"\n❌ Tutorial failed: {e}")
            print("Please check your setup and try again.")

def main():
    """
    Main function to run the tutorial.
    
    Before running this tutorial, make sure you have:
    1. Installed required packages: pip install haystack-ai python-dotenv
    2. Created a .env file with your OpenAI API key
    3. Python 3.8 or higher
    """
    
    # Create and run the tutorial
    tutorial = HaystackTutorial()
    tutorial.run_complete_tutorial()

# Example of how to use individual components (advanced users)
def advanced_examples():
    """
    Advanced examples showing specific Haystack features.
    These examples show more advanced concepts after you understand the basics.
    """
    print("\n🔬 ADVANCED EXAMPLES")
    print("=" * 30)
    
    # Example 1: Custom document creation
    print("\n1. Creating custom documents:")
    custom_docs = [
        Document(content="Haystack is an open-source framework for building AI applications."),
        Document(content="It supports various LLMs and vector databases."),
        Document(content="RAG systems combine retrieval and generation for better AI answers.")
    ]
    
    for i, doc in enumerate(custom_docs, 1):
        print(f"   Document {i}: {doc.content}")
    
    # Example 2: Different embedding models
    print("\n2. Different embedding models you can use:")
    embedding_models = [
        "text-embedding-3-small",  # OpenAI (efficient)
        "text-embedding-3-large",  # OpenAI (more accurate)
        "text-embedding-ada-002"   # OpenAI (legacy)
    ]
    
    for model in embedding_models:
        print(f"   - {model}")
    
    # Example 3: Pipeline customization tips
    print("\n3. Pipeline customization tips:")
    tips = [
        "Adjust top_k in retrievers to get more/fewer documents",
        "Use different splitting strategies (word, sentence, passage)",
        "Experiment with embedding models for better search quality",
        "Customize prompt templates for specific use cases",
        "Add preprocessing components for data cleaning"
    ]
    
    for tip in tips:
        print(f"   • {tip}")

if __name__ == "__main__":
    main()
    
    # Uncomment to see advanced examples
    # advanced_examples()