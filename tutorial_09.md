# LlamaIndex RAG Tutorial: A Complete Beginner's Guide

## Table of Contents
1. [What We're Building](#what-were-building)
2. [Prerequisites](#prerequisites)
3. [Understanding the Code Structure](#understanding-the-code-structure)
4. [Step-by-Step Code Walkthrough](#step-by-step-code-walkthrough)
5. [Running the Tutorial](#running-the-tutorial)
6. [Understanding What Happens Behind the Scenes](#understanding-what-happens-behind-the-scenes)
7. [Troubleshooting](#troubleshooting)

---

## What We're Building

Imagine you have a pile of documents and you want to ask questions about them, just like asking a librarian. Our RAG system will:

1. **Read your documents** (like a librarian reading books)
2. **Remember everything** in a smart way (like creating a super-detailed catalog)
3. **Answer your questions** by finding relevant information and explaining it

**Example:**
- You have company documents
- You ask: "What services does the company provide?"
- The system finds the relevant parts and gives you a clear answer

---

## Prerequisites

### What You Need to Install
```bash
# Install the required packages
pip install llama-index python-dotenv
```

### What You Need to Set Up
1. **OpenAI API Key**: Get one from [OpenAI's website](https://platform.openai.com/)
2. **Create a .env file** in your project folder:
```
OPENAI_API_KEY=your_actual_api_key_here
```

---

## Understanding the Code Structure

Our tutorial is organized into logical functions. Think of each function as a specific job:

```
setup_environment()       → Set up API keys and connections
create_sample_data()      → Create example documents to work with
load_and_index_documents() → Read documents and make them searchable
save_index()              → Save our work so we don't have to redo it
load_existing_index()     → Load previously saved work
query_system()            → The interactive part where you ask questions
```

---

## Step-by-Step Code Walkthrough

### Step 1: Setting Up the Environment

```python
def setup_environment():
    """Load environment variables and set up OpenAI"""
    load_dotenv()  # This reads your .env file
    
    # Get your API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("❌ No OPENAI_API_KEY found in .env file.")
        exit(1)
    
    # Tell LlamaIndex to use OpenAI's GPT model
    Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
```

**What this does in plain English:**
- Reads your secret API key from the .env file
- Checks if the key exists (stops if it doesn't)
- Tells LlamaIndex to use GPT-3.5-turbo as the "brain" for answering questions

### Step 2: Creating Sample Documents

```python
def create_sample_data():
    """Create sample documents if they don't exist"""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)  # Create 'data' folder
    
    # Create a text file with company information
    doc1_path = data_dir / "company_info.txt"
    if not doc1_path.exists():
        with open(doc1_path, "w") as f:
            f.write("About Our Company\n=================\n...")
```

**What this does in plain English:**
- Creates a folder called "data"
- Writes two sample text files with fake company information
- Only creates files if they don't already exist
- Returns the folder location so other functions can find the files

### Step 3: Loading and Indexing Documents

```python
def load_and_index_documents(data_dir):
    """Load documents and create searchable index"""
    # Read all documents from the folder
    documents = SimpleDirectoryReader(str(data_dir)).load_data()
    
    # Convert documents into a searchable format
    index = VectorStoreIndex.from_documents(documents)
    
    return index
```

**What this does in plain English:**
- Reads every text file in your data folder
- Converts the text into mathematical representations (vectors) that computers can search through quickly
- Think of it like creating a super-detailed index at the back of a book, but much smarter

### Step 4: Saving Your Work

```python
def save_index(index, persist_dir="./storage"):
    """Save index to disk to avoid re-creating it"""
    index.storage_context.persist(persist_dir=persist_dir)
```

**What this does in plain English:**
- Saves all the processed data to a "storage" folder
- Next time you run the program, you won't have to process everything again
- Like saving your progress in a video game

### Step 5: Loading Existing Work

```python
def load_existing_index(persist_dir="./storage"):
    """Load existing index from disk"""
    try:
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
        return index
    except:
        return None  # No saved work found
```

**What this does in plain English:**
- Checks if you have previously saved work
- If yes, loads it up (much faster than starting over)
- If no, returns None so the program knows to create everything from scratch

### Step 6: The Interactive Question-Answering System

```python
def query_system(index):
    """Interactive querying system"""
    query_engine = index.as_query_engine()
    
    while True:
        question = input("❓ Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        response = query_engine.query(question)
        print(f"\n🎯 Answer: {response}\n")
```

**What this does in plain English:**
- Creates a "query engine" - think of it as your personal research assistant
- Starts a loop where you can keep asking questions
- For each question:
  1. Finds relevant information from your documents
  2. Uses GPT to understand the question and information
  3. Gives you a clear, helpful answer
- Type 'quit' to stop

### Step 7: The Main Function (Orchestra Conductor)

```python
def main():
    """Main function that orchestrates the RAG system"""
    # 1. Set up the environment
    setup_environment()
    
    # 2. Make sure we have documents to work with
    data_dir = create_sample_data()
    
    # 3. Try to load existing work
    index = load_existing_index()
    
    # 4. If no existing work, create new index
    if index is None:
        index = load_and_index_documents(data_dir)
        save_index(index)
    
    # 5. Start the interactive session
    query_system(index)
```

**What this does in plain English:**
- Acts like a conductor of an orchestra, making sure everything happens in the right order
- Sets up everything you need
- Either loads your previous work or creates new work
- Starts the interactive question-answering session

---

## Running the Tutorial

### Method 1: Run the Full Interactive Tutorial
```bash
python script_09.py
```

**What you'll see:**
```
🚀 Welcome to LlamaIndex RAG Tutorial!
==================================================
✅ Environment setup complete!
✅ Sample data created in data
🔍 Creating vector index (this may take a moment)...
✅ Vector index created successfully!
💾 Saving index to ./storage...
✅ Index saved!

🤖 RAG System Ready! Ask questions about the loaded documents.
Type 'quit' to exit

❓ Your question: 
```

### Method 2: Try Example Questions
Once the system is running, try these questions:

1. **"What services does TechCorp provide?"**
   - Expected: List of AI consulting, web development, mobile apps, etc.

2. **"How many engineers work at the company?"**
   - Expected: Breakdown of 20 software engineers, 8 AI/ML engineers, etc.

3. **"Who is the CEO?"**
   - Expected: Sarah Johnson with 15 years experience

4. **"Tell me about the team structure"**
   - Expected: Overview of engineering, design, and business teams

---

## Understanding What Happens Behind the Scenes

### When You Ask a Question

1. **Your Question**: "What services does TechCorp provide?"

2. **The System Searches**: 
   - Converts your question into a mathematical format
   - Searches through all document chunks for relevant information
   - Finds the section about "Key Services"

3. **Context Building**:
   - Takes the relevant chunks
   - Creates a prompt like: "Based on this information: [relevant text], answer the user's question: What services does TechCorp provide?"

4. **AI Response**:
   - GPT reads the context and your question
   - Generates a clear, accurate answer based only on your documents

### The Magic of Vector Search

Think of vector search like this:
- Every sentence gets converted to a "fingerprint" (vector)
- Similar sentences have similar fingerprints
- When you ask a question, it finds sentences with similar fingerprints
- Much smarter than just keyword matching!

---

## Troubleshooting

### Common Issues and Solutions

#### 1. "No OPENAI_API_KEY found"
**Problem**: Your .env file isn't set up correctly
**Solution**: 
- Create a file named `.env` (not .env.txt)
- Add: `OPENAI_API_KEY=your_actual_key_here`
- Make sure there are no spaces around the = sign

#### 2. "Module not found" errors
**Problem**: Packages aren't installed
**Solution**: 
```bash
pip install llama-index python-dotenv
```

#### 3. Slow response times
**Problem**: First-time setup takes time
**Solution**: 
- First run is always slower (creating the index)
- Subsequent runs are much faster (loading saved index)
- The actual querying should be quick

#### 4. Empty or weird responses
**Problem**: Documents might not contain relevant information
**Solution**:
- Make sure your documents contain information related to your questions
- Try asking about topics you know are in the sample documents

### File Structure After Running
```
your_project/
├── your_tutorial_file.py
├── .env
├── data/
│   ├── company_info.txt
│   └── team_info.txt
└── storage/
    ├── docstore.json
    ├── index_store.json
    └── vector_store.json
```

---

## Next Steps

Once you understand this tutorial:

1. **Replace sample data**: Put your own documents in the `data` folder
2. **Try different file types**: LlamaIndex can read PDFs, Word docs, etc.
3. **Experiment with questions**: See how well it handles different types of queries
4. **Learn about agents**: Explore the more advanced agent examples in the code

This tutorial gives you a solid foundation for building more complex RAG systems with LlamaIndex!