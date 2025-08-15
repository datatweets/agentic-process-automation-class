# Haystack Tutorial for Beginners: Building Your First AI Question-Answering System

## Table of Contents
1. [What We're Building](#what-were-building)
2. [Understanding the Basics](#understanding-the-basics)
3. [Setup and Installation](#setup-and-installation)
4. [Code Walkthrough](#code-walkthrough)
5. [Running the Tutorial](#running-the-tutorial)
6. [Understanding the Results](#understanding-the-results)
7. [What Happens Behind the Scenes](#what-happens-behind-the-scenes)
8. [Common Issues and Solutions](#common-issues-and-solutions)

---

## What We're Building

Imagine you have a pile of company documents and you want to build a system where employees can ask questions like:
- "How many vacation days do I get?"
- "What's our remote work policy?"
- "How do I authenticate with our API?"

Instead of manually searching through documents, our AI system will:
1. **Read and understand** all your documents
2. **Find relevant information** when someone asks a question
3. **Generate a clear answer** using AI

This is called a **RAG system** (Retrieval-Augmented Generation).

---

## Understanding the Basics

### 🧩 What are Components?
Think of components as specialized workers in a factory:
- **Embedder**: Converts text into numbers that computers can understand
- **Retriever**: Searches through documents to find relevant information
- **Generator**: Creates human-like answers using AI
- **Document Store**: A smart filing cabinet that stores all your documents

### 🔗 What are Pipelines?
Pipelines are like assembly lines that connect these workers:
```
Document → Process → Store → Search → Generate Answer
```

### 🤖 What is RAG?
RAG has two steps:
1. **Retrieval**: Find relevant documents for the question
2. **Generation**: Use AI to create an answer based on those documents

---

## Setup and Installation

### Step 1: Install Required Packages
```bash
pip install haystack-ai python-dotenv
```

### Step 2: Get an OpenAI API Key
1. Go to [OpenAI's website](https://platform.openai.com/)
2. Create an account and get an API key
3. Keep this key safe - you'll need it!

### Step 3: Create Environment File
Create a file called `.env` in your project folder:
```
OPENAI_API_KEY=your_actual_api_key_here
```

**Important**: Replace `your_actual_api_key_here` with your real API key!

---

## Code Walkthrough

Let's go through the code step by step, explaining what each part does in simple terms.

### The Main Class Structure

```python
class HaystackTutorial:
    def __init__(self):
        self.setup_environment()
        self.document_store = None
        self.indexing_pipeline = None
        self.search_pipeline = None
        self.rag_pipeline = None
```

**What this does**: Creates a tutorial class that will hold all our components and pipelines.

### Step 1: Environment Setup

```python
def setup_environment(self):
    load_dotenv()  # Read the .env file
    self.api_key = os.getenv("OPENAI_API_KEY")
    if not self.api_key:
        raise ValueError("No API key found!")
```

**What this does**: 
- Reads your API key from the .env file
- Checks if the key exists
- Stops the program if no key is found

### Step 2: Creating Sample Documents

```python
def create_sample_documents(self):
    data_dir = Path("tutorial_data")
    data_dir.mkdir(exist_ok=True)
    
    documents_content = {
        "company_handbook.txt": "VACATION POLICY: 15 days per year...",
        "technical_documentation.txt": "API AUTHENTICATION: Bearer tokens...",
        "project_guidelines.txt": "SPRINT PLANNING: 2 weeks long..."
    }
```

**What this does**:
- Creates a folder called "tutorial_data"
- Creates three sample text files with realistic company information
- These represent the documents your AI will learn from

### Step 3: Document Store Setup

```python
def setup_document_store(self):
    self.document_store = InMemoryDocumentStore()
```

**What this does**:
- Creates a "smart filing cabinet" in your computer's memory
- This will store all documents in a searchable format
- **Note**: Data is lost when program ends (for learning purposes)

### Step 4: Creating the Indexing Pipeline

This is where the magic begins! The indexing pipeline processes your documents to make them searchable.

```python
def create_indexing_pipeline(self):
    # Component 1: Convert files to Document objects
    file_converter = TextFileToDocument()
    
    # Component 2: Split documents into smaller chunks
    document_splitter = DocumentSplitter(
        split_by="sentence",
        split_length=3,
        split_overlap=1
    )
    
    # Component 3: Create embeddings
    document_embedder = OpenAIDocumentEmbedder(
        model="text-embedding-3-small"
    )
    
    # Component 4: Store documents
    document_writer = DocumentWriter(document_store=self.document_store)
```

**What each component does**:
1. **TextFileToDocument**: Reads text files and converts them to Document objects
2. **DocumentSplitter**: Breaks large documents into smaller, manageable pieces
3. **OpenAIDocumentEmbedder**: Converts text to mathematical representations (vectors)
4. **DocumentWriter**: Saves everything to the document store

**Why split documents?**
- Large documents are hard to search through
- Small chunks give more precise answers
- AI models have limits on how much text they can process

### Step 5: Connecting the Pipeline

```python
# Create the pipeline
self.indexing_pipeline = Pipeline()

# Add components
self.indexing_pipeline.add_component("file_converter", file_converter)
self.indexing_pipeline.add_component("splitter", document_splitter)
self.indexing_pipeline.add_component("embedder", document_embedder)
self.indexing_pipeline.add_component("writer", document_writer)

# Connect components (data flows from one to the next)
self.indexing_pipeline.connect("file_converter", "splitter")
self.indexing_pipeline.connect("splitter", "embedder")
self.indexing_pipeline.connect("embedder", "writer")
```

**What this does**:
- Creates an assembly line: File → Split → Embed → Store
- Data flows automatically from one component to the next

### Step 6: Running the Indexing

```python
def index_documents(self, data_dir):
    file_paths = list(data_dir.glob("*.txt"))
    
    result = self.indexing_pipeline.run({
        "file_converter": {"sources": file_paths}
    })
```

**What this does**:
- Finds all .txt files in the data directory
- Runs them through the indexing pipeline
- Now your documents are searchable!

### Step 7: Creating the Search Pipeline

```python
def create_search_pipeline(self):
    # Convert question to numbers
    query_embedder = OpenAITextEmbedder(
        model="text-embedding-3-small"
    )
    
    # Search for similar documents
    retriever = InMemoryEmbeddingRetriever(
        document_store=self.document_store,
        top_k=3
    )
    
    # Connect them
    self.search_pipeline = Pipeline()
    self.search_pipeline.add_component("query_embedder", query_embedder)
    self.search_pipeline.add_component("retriever", retriever)
    self.search_pipeline.connect("query_embedder.embedding", "retriever.query_embedding")
```

**What this does**:
- Converts your question to the same mathematical format as documents
- Searches for the 3 most similar document chunks
- This is the "Retrieval" part of RAG

### Step 8: Creating the RAG Pipeline

This combines search with AI generation to create answers.

```python
def create_rag_pipeline(self):
    prompt_template = """
    Answer the question based on the given context.
    
    Context:
    {% for document in documents %}
    {{ document.content }}
    {% endfor %}
    
    Question: {{ question }}
    
    Answer: Provide a clear, accurate answer based only on the information above.
    """
```

**What the template does**:
- Tells the AI how to use the retrieved documents
- Provides structure for generating answers
- Ensures answers are based on your documents, not general knowledge

---

## Running the Tutorial

### Method 1: Run the Complete Tutorial
```bash
python haystack_tutorial.py
```

### Method 2: Run Individual Steps
You can also run parts of the tutorial separately to understand each component.

### What You'll See

When you run the tutorial, you'll see output like this:

```
🚀 STEP 1: Setting up environment...
✅ Environment setup complete!
✅ API key found: sk-proj-Ab...

📚 STEP 2: Creating sample documents...
✅ Created: company_handbook.txt
✅ Created: technical_documentation.txt
✅ Created: project_guidelines.txt

🗃️ STEP 3: Setting up document store...
✅ Document store created!

🔄 STEP 4: Creating indexing pipeline...
✅ Indexing pipeline created!

📥 STEP 5: Indexing documents...
✅ Successfully indexed 15 document chunks!

🔍 STEP 6: Creating search pipeline...
✅ Search pipeline created!

🤖 STEP 7: Creating RAG pipeline...
✅ RAG pipeline created!
```

---

## Understanding the Results

### Search Results Example
When you search for "vacation policy":
```
🔎 Searching for: 'vacation policy'
📄 Found 3 relevant documents:

Document 1:
Score: 0.892
Content: VACATION POLICY: All full-time employees are entitled to 15 days of paid vacation per year. Vacation time must be requested at least 2 weeks in advance...
```

**What the score means**: Higher scores (closer to 1.0) mean the document is more relevant to your question.

### Q&A Results Example
When you ask "How many vacation days do employees get?":
```
❓ Question: How many vacation days do employees get?
🎯 Answer: All full-time employees are entitled to 15 days of paid vacation per year.
📚 Based on 3 relevant documents
```

---

## What Happens Behind the Scenes

### 1. When You Index Documents
```
Text File → Document Object → Small Chunks → Numbers (Embeddings) → Document Store
```

**Example**:
- Original: "Employees get 15 vacation days"
- Becomes: [0.2, -0.1, 0.8, 0.3, ...] (thousands of numbers)

### 2. When You Ask a Question
```
Question → Numbers (Embedding) → Search Document Store → Find Similar → Generate Answer
```

**Example**:
- Question: "How many vacation days?"
- Becomes: [0.1, -0.2, 0.7, 0.4, ...] (similar pattern to vacation documents)
- Finds: Documents about vacation policy
- Generates: "Employees get 15 vacation days per year"

### 3. Why This Works
- Similar concepts have similar number patterns
- Computers can quickly compare these patterns
- AI understands context and generates human-like responses

---

## Common Issues and Solutions

### Issue 1: "No API key found"
**Problem**: Your .env file isn't set up correctly
**Solution**: 
```bash
# Create .env file (not .env.txt!)
echo "OPENAI_API_KEY=your_actual_key" > .env
```

### Issue 2: "Module not found" errors
**Problem**: Packages aren't installed
**Solution**:
```bash
pip install haystack-ai python-dotenv
```

### Issue 3: Slow responses
**Problem**: First run takes time
**Solution**: 
- First run is always slower (creating embeddings)
- Subsequent runs are faster
- Be patient during indexing step

### Issue 4: Poor answer quality
**Problem**: AI gives vague or wrong answers
**Solutions**:
- Add more detailed documents
- Adjust the prompt template
- Increase `top_k` to retrieve more documents

### Issue 5: "Rate limit exceeded"
**Problem**: Too many API calls to OpenAI
**Solution**:
- Wait a few minutes
- Check your OpenAI billing status
- Reduce the number of test questions

---

## Teaching Tips for Instructors

### 1. Start with Analogies
- **Document Store**: Like a smart library catalog
- **Embeddings**: Like GPS coordinates for meaning
- **Pipeline**: Like an assembly line in a factory
- **RAG**: Like a research assistant who reads first, then answers

### 2. Show Visual Progress
- Run each step separately to show progress
- Print intermediate results to demonstrate what's happening
- Use the search demo before Q&A to show retrieval working

### 3. Common Student Questions

**"Why not just use Google?"**
- This searches YOUR documents, not the internet
- Gives answers based on YOUR company's specific information
- More accurate for internal knowledge

**"Why split documents into chunks?"**
- AI models have text limits
- Smaller chunks give more precise answers
- Better search accuracy

**"What are embeddings?"**
- Mathematical representations of meaning
- Similar concepts have similar numbers
- Allows semantic (meaning-based) search, not just keyword matching

### 4. Hands-on Exercises
1. Have students modify the sample documents
2. Try different questions and observe results
3. Experiment with different `top_k` values
4. Modify the prompt template

### 5. Real-world Applications
- Customer support chatbots
- Internal company knowledge bases
- Legal document analysis
- Medical research assistance
- Educational Q&A systems

---

## Next Steps for Students

After completing this tutorial, students can:

1. **Replace sample data**: Use their own documents
2. **Try different file types**: PDFs, Word docs, etc.
3. **Experiment with parameters**: Chunk sizes, embedding models
4. **Add persistence**: Use Elasticsearch or other databases
5. **Build a web interface**: Flask/FastAPI frontend
6. **Add authentication**: Secure access controls

This tutorial provides a solid foundation for understanding modern AI document processing and retrieval systems!