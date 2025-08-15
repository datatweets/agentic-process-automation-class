"""
LlamaIndex RAG Tutorial for Beginners
=====================================

This tutorial demonstrates how to build a simple RAG (Retrieval-Augmented Generation) 
system using LlamaIndex. The system will load documents, create a searchable index,
and answer questions based on your data.

Prerequisites:
- pip install llama-index
- pip install python-dotenv
- OpenAI API key in .env file
"""

import os
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

# LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.llms.openai import OpenAI

# Setup logging to see what's happening
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

def setup_environment():
    """Load environment variables and set up OpenAI"""
    load_dotenv()
    
    # Get OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("❌ No OPENAI_API_KEY found in .env file.")
        print("Please add: OPENAI_API_KEY=your_api_key_here")
        exit(1)
    
    # Configure LlamaIndex to use OpenAI
    Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
    print("✅ Environment setup complete!")

def create_sample_data():
    """Create sample documents if they don't exist"""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Sample document 1
    doc1_path = data_dir / "company_info.txt"
    if not doc1_path.exists():
        with open(doc1_path, "w") as f:
            f.write("""
            About Our Company
            =================
            
            TechCorp is a leading software development company founded in 2010.
            We specialize in AI solutions, web development, and mobile applications.
            
            Our mission is to make technology accessible to everyone through 
            innovative and user-friendly solutions.
            
            Key Services:
            - Artificial Intelligence consulting
            - Custom web application development  
            - Mobile app development (iOS and Android)
            - Cloud infrastructure setup
            - Data analytics and visualization
            
            We have successfully completed over 500 projects for clients 
            ranging from startups to Fortune 500 companies.
            """)
    
    # Sample document 2
    doc2_path = data_dir / "team_info.txt"
    if not doc2_path.exists():
        with open(doc2_path, "w") as f:
            f.write("""
            Our Team
            ========
            
            TechCorp has a diverse team of 50+ professionals:
            
            Engineering Team:
            - 20 Software Engineers (Python, JavaScript, Java)
            - 8 AI/ML Engineers (TensorFlow, PyTorch)
            - 5 DevOps Engineers (AWS, Docker, Kubernetes)
            
            Design Team:
            - 6 UX/UI Designers
            - 3 Graphic Designers
            
            Business Team:
            - 4 Project Managers
            - 3 Business Analysts
            - 2 Sales Representatives
            
            Leadership:
            - CEO: Sarah Johnson (15 years experience)
            - CTO: Michael Chen (12 years experience)
            - Head of Design: Emma Rodriguez (10 years experience)
            """)
    
    print(f"✅ Sample data created in {data_dir}")
    return data_dir

def load_and_index_documents(data_dir):
    """Load documents and create searchable index"""
    print("📚 Loading documents...")
    
    # Load documents from the data directory
    documents = SimpleDirectoryReader(str(data_dir)).load_data()
    print(f"✅ Loaded {len(documents)} documents")
    
    # Create vector index from documents
    print("🔍 Creating vector index (this may take a moment)...")
    index = VectorStoreIndex.from_documents(documents)
    print("✅ Vector index created successfully!")
    
    return index

def save_index(index, persist_dir="./storage"):
    """Save index to disk to avoid re-creating it"""
    print(f"💾 Saving index to {persist_dir}...")
    index.storage_context.persist(persist_dir=persist_dir)
    print("✅ Index saved!")

def load_existing_index(persist_dir="./storage"):
    """Load existing index from disk"""
    try:
        print(f"📂 Loading existing index from {persist_dir}...")
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
        print("✅ Existing index loaded!")
        return index
    except:
        print("ℹ️  No existing index found, will create new one")
        return None

def query_system(index):
    """Interactive querying system"""
    print("\n🤖 RAG System Ready! Ask questions about the loaded documents.")
    print("Type 'quit' to exit\n")
    
    # Create query engine
    query_engine = index.as_query_engine()
    
    while True:
        # Get user question
        question = input("❓ Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not question:
            continue
        
        try:
            print("🔄 Searching for relevant information...")
            
            # Query the index
            response = query_engine.query(question)
            
            print(f"\n🎯 Answer: {response}\n")
            print("-" * 60)
            
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main function that orchestrates the RAG system"""
    print("🚀 Welcome to LlamaIndex RAG Tutorial!")
    print("=" * 50)
    
    # Step 1: Setup environment
    setup_environment()
    
    # Step 2: Create or ensure sample data exists
    data_dir = create_sample_data()
    
    # Step 3: Check if we have an existing index
    persist_dir = "./storage"
    index = load_existing_index(persist_dir)
    
    # Step 4: If no existing index, create one
    if index is None:
        index = load_and_index_documents(data_dir)
        save_index(index, persist_dir)
    
    # Step 5: Start interactive querying
    query_system(index)

# Example standalone functions for specific use cases
def simple_query_example():
    """Simple example of querying without interactive loop"""
    setup_environment()
    data_dir = create_sample_data()
    
    # Load and index
    documents = SimpleDirectoryReader(str(data_dir)).load_data()
    index = VectorStoreIndex.from_documents(documents)
    
    # Create query engine
    query_engine = index.as_query_engine()
    
    # Ask a question
    response = query_engine.query("What services does TechCorp provide?")
    print(f"Response: {response}")

def advanced_agent_example():
    """Example of using LlamaIndex with agents (like in the images)"""
    setup_environment()
    
    # This would be similar to the agent setup shown in the images
    from llama_index.core.agent import FunctionCallingAgentWorker, AgentRunner
    from llama_index.llms.openai import OpenAI
    
    # Setup LLM
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # Create sample tools (vector_tool and summary_tool would be defined)
    # This is a simplified version - in practice you'd have actual tools
    print("🔧 Advanced agent example - see documentation for full implementation")

if __name__ == "__main__":
    # Run the main tutorial
    main()
    
    # Uncomment to run simple example instead:
    # simple_query_example()