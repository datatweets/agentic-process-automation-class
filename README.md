# Agentic AI Examples

This repository contains a collection of Python scripts demonstrating various implementations of AI agents and tools using LangChain, OpenAI GPT models, and other APIs.

## 🚀 Features

- Interactive AI Assistant with Weather and Wikipedia capabilities
- DuckDuckGo Search Integration
- Custom AI Agents with Memory
- Weather API Integration
- Wikipedia Information Retrieval
- Conversational Memory
- Custom Tool Creation

## 📋 Prerequisites

```bash
python 3.9+
pip
```

## 🛠 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd agenticai
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with your API keys:
```plaintext
OPENAI_API_KEY=your_openai_api_key
SERPAPI_API_KEY=your_serpapi_api_key
```

## 📁 Project Structure

- `script_01.py` - Initial setup and basic configuration
- `script_02.py` - Weather and Wikipedia information retrieval using OpenAI functions
- `script_03.py` - Additional functionalities
- `script_04.py` - DuckDuckGo search integration
- `script_05.py` - Advanced AI agent with custom tools and memory

## 🔍 Scripts Overview

### script_02.py - Weather & Wikipedia Assistant
- Uses OpenAI's function calling feature
- Integrates with weather API and Wikipedia
- Interactive command-line interface

### script_04.py - DuckDuckGo Search
- Implements DuckDuckGo search functionality
- Uses LangChain for query optimization
- Includes retry strategies and error handling

### script_05.py - Advanced AI Agent
Features:
- Google Search via SerpAPI
- LLM Math capabilities
- Custom time tool
- Conversational memory
- Interactive chat interface

## 🚀 Usage

### Weather & Wikipedia Assistant (script_02.py)
```bash
python script_02.py
# Example queries:
# "What's the weather in Paris?"
# "Tell me about quantum physics"
```

### DuckDuckGo Search (script_04.py)
```bash
python script_04.py
# Enter your search query when prompted
```

### AI Agent with Memory (script_05.py)
```bash
python script_05.py
# Example queries:
# "What year was Einstein born? What is that year multiplied by 5?"
# "What time is it?"
# "Suggest Thai food recipes"
```

## 📚 API Documentation

### Weather API
- Uses Open-Meteo API for weather data
- Endpoint: https://api.open-meteo.com/v1/forecast
- Free to use, no API key required

### Wikipedia API
- Uses MediaWiki API for Wikipedia information
- Endpoint: https://en.wikipedia.org/w/api.php
- Free to use, no API key required

### Search APIs
- DuckDuckGo API (script_04.py)
- SerpAPI (script_05.py, requires API key)

## ⚙️ Configuration

The project uses environment variables for configuration. Create a `.env` file with:

```plaintext
OPENAI_API_KEY=your_openai_api_key
SERPAPI_API_KEY=your_serpapi_api_key  # Only needed for script_05.py
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📮 Contact

For questions and support, please open an issue in the repository.

## 🙏 Acknowledgments

- OpenAI for their GPT models
- LangChain for their framework
- Open-Meteo for weather data
- Wikipedia for information access
