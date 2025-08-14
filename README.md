# Agentic Process Automation Class

Welcome to the **Agentic Process Automation Class** repository! This comprehensive collection contains tutorials and examples for building AI agents that can automate processes using LangGraph, LangChain, and various LLM providers.

## 🚀 What You'll Learn

This repository provides step-by-step tutorials covering:

- **Process automation fundamentals** - Understanding how AI agents can automate workflows
- **Agent architectures** - Building ReAct agents, chatbots, and tool-using agents  
- **Memory and persistence** - Creating agents that remember context across sessions
- **Web search integration** - Using Tavily for AI-powered information gathering
- **Multi-LLM support** - Working with OpenAI, Groq, and other providers
- **Real-world automation** - Practical examples for process automation scenarios

## 📁 Repository Structure

```
agentic-process-automation-class/
├── script_01.py          # Basic LangGraph introduction
├── script_02.py          # Weather & Wikipedia assistant with function calling
├── script_03.py          # ReAct agent with custom tools
├── script_04.py          # DuckDuckGo search integration
├── script_05.py          # Advanced AI agent with memory and multiple tools
├── script_06.py          # Chatbot + ReAct agent demos (Groq/OpenAI)
├── script_07.py          # Tavily search integration for web research
├── script_08.py          # Persistent chatbot with conversation threads
├── tutorial_02.md        # Weather & Wikipedia tutorial
├── tutorial_03.md        # ReAct agent tutorial
├── tutorial_05.md        # Advanced agent tutorial
├── tutorial_06.md        # LangGraph chatbot and ReAct tutorial
├── tutorial_07.md        # Tavily search tutorial
├── tutorial_08.md        # Persistent memory tutorial
├── requirements.txt      # Python dependencies
├── .env.sample          # Environment variables template
├── .gitignore           # Git ignore configuration
└── README.md            # This file
```

## 📋 Prerequisites

- Python 3.9+
- pip package manager
- API keys for:
  - OpenAI (required for most scripts)
  - Groq (optional, for alternative LLM)
  - Tavily (for web search functionality)
  - SerpAPI (for Google search integration)

## 🛠 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/datatweets/agentic-process-automation-class.git
cd agentic-process-automation-class
```

2. **Create a virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**
```bash
cp .env.sample .env
# Edit .env and add your API keys
```

## 🔧 Environment Setup

Create a `.env` file with your API keys:

```plaintext
# Required for most scripts
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Alternative LLM provider
GROQ_API_KEY=your_groq_api_key_here

# For web search functionality (script_07.py)
TAVILY_API_KEY=your_tavily_api_key_here

# For Google search (script_05.py)
SERPAPI_API_KEY=your_serpapi_api_key_here
```

## 📚 Script Descriptions

### script_01.py - Basic LangGraph Introduction
Basic setup and LangGraph fundamentals.

### script_02.py - Weather & Wikipedia Assistant
- OpenAI function calling
- Weather API integration
- Wikipedia information retrieval
- **Tutorial:** `tutorial_02.md`

### script_03.py - ReAct Agent with Tools
- ReAct (Reasoning + Acting) pattern
- Custom tool creation
- Agent decision-making
- **Tutorial:** `tutorial_03.md`

### script_04.py - DuckDuckGo Search Integration
- Web search capabilities
- Query optimization
- Error handling and retries

### script_05.py - Advanced Multi-Tool Agent
- Google Search via SerpAPI
- Math calculations
- Time utilities
- Conversational memory
- **Tutorial:** `tutorial_05.md`

### script_06.py - Chatbot & ReAct Demos
- Simple StateGraph chatbot
- Prebuilt ReAct agent
- Multi-LLM support (OpenAI/Groq)
- **Tutorial:** `tutorial_06.md`

### script_07.py - Tavily Web Search
- Direct Q&A with web search
- Structured search results
- Intelligent search agents
- **Tutorial:** `tutorial_07.md`

### script_08.py - Persistent Memory Chatbot
- Conversation persistence
- Thread-based memory isolation
- Multi-user support
- **Tutorial:** `tutorial_08.md`

## 🚀 Quick Start

1. **Basic chatbot (script_06.py):**
```bash
python script_06.py chat
```

2. **Web search agent (script_07.py):**
```bash
python script_07.py
```

3. **Persistent memory bot (script_08.py):**
```bash
python script_08.py
```

## 🔍 Example Use Cases

### Customer Support Automation
Use persistent memory chatbots to handle customer inquiries with context retention.

### Research Automation
Combine web search with ReAct agents for automated research tasks.

### Data Collection
Build agents that can search, extract, and process information from multiple sources.

### Workflow Orchestration
Create multi-step automation workflows using LangGraph's state management.

## 📖 Learning Path

1. **Start with basics:** `script_02.py` → `tutorial_02.md`
2. **Learn ReAct pattern:** `script_03.py` → `tutorial_03.md`
3. **Add complexity:** `script_05.py` → `tutorial_05.md`
4. **Explore LangGraph:** `script_06.py` → `tutorial_06.md`
5. **Web integration:** `script_07.py` → `tutorial_07.md`
6. **Persistence:** `script_08.py` → `tutorial_08.md`

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain & LangGraph** - For the agent framework
- **OpenAI** - For GPT models and APIs
- **Groq** - For fast open-source model inference
- **Tavily** - For AI-optimized web search
- **Open-Meteo** - For weather data API

## 📮 Support

For questions, issues, or contributions:
- Open an issue in this repository
- Check existing tutorials for guidance
- Review the comprehensive documentation in each tutorial

---

**Happy Learning! 🚀**

Start building intelligent process automation with AI agents today!