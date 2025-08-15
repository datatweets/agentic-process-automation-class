"""
Multi-Agent Stock Analysis System using AutoGen
===============================================

A simple multi-agent system for stock analysis with:
- User Proxy Agent (handles human interaction and code execution)
- Assistant Agent (generates Python code for analysis)

Prerequisites:
- pip install autogen-agentchat yfinance matplotlib pandas python-dotenv
- Create a .env file with: OPENAI_API_KEY=your_api_key_here
"""

import os
from dotenv import load_dotenv
from autogen import ConversableAgent, UserProxyAgent

# Load environment variables from .env file
load_dotenv()

class StockAnalysisSystem:
    """Multi-Agent Stock Analysis System with User Proxy and Assistant agents."""
    
    def __init__(self):
        """Initialize the multi-agent system."""
        self.validate_api_key()
        self.setup_config()
        self.create_agents()
        
    def validate_api_key(self):
        """Check if OpenAI API key is available."""
        if not os.getenv("OPENAI_API_KEY"):
            print("❌ Error: OpenAI API key not found!")
            print("Please create a .env file with:")
            print("OPENAI_API_KEY=your_api_key_here")
            exit(1)
        
    def setup_config(self):
        """Configure the LLM settings for AutoGen."""
        self.config_list = [{
            "model": "gpt-4o-mini",
            "api_key": os.getenv("OPENAI_API_KEY"),
        }]
        
        self.llm_config = {
            "config_list": self.config_list,
            "temperature": 0.1,
            "timeout": 60,
        }
        
    def create_agents(self):
        """Create the User Proxy Agent and Assistant Agent."""
        
        # User Proxy Agent - handles human interaction and code execution
        self.user_proxy = UserProxyAgent(
            name="UserProxy",
            system_message="""You are a User Proxy Agent for stock analysis.
            - Execute Python code provided by the Assistant Agent
            - Handle package installation if needed
            - Provide feedback on outputs
            - Ask for human input when needed""",
            
            human_input_mode="ALWAYS",
            max_consecutive_auto_reply=3,
            is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),
            
            code_execution_config={
                "work_dir": "stock_analysis",
                "use_docker": False,
            },
            llm_config=self.llm_config
        )
        
        # Assistant Agent - generates Python code for stock analysis
        self.assistant = ConversableAgent(
            name="AssistantAgent", 
            system_message="""You are an expert Python programmer for financial data analysis.
            
            Your tasks:
            1. Write Python code for stock analysis and visualization
            2. Use yfinance for data fetching
            3. Create matplotlib visualizations
            4. Handle errors gracefully
            5. Install required packages when needed
            
            Code requirements:
            - Always import required libraries
            - Include error handling
            - Add helpful comments
            - Create clear visualizations""",
            
            llm_config=self.llm_config,
            human_input_mode="NEVER",
        )
    
    def start_analysis(self, request=None):
        """Start the multi-agent stock analysis conversation."""
        if not request:
            try:
                request = input("Enter your stock analysis request: ").strip()
                if not request:
                    print("❌ No request provided.")
                    return None
            except (EOFError, KeyboardInterrupt):
                print("❌ Input not available. Use start_analysis('your request') directly.")
                return None
        
        print("🚀 Starting Multi-Agent Stock Analysis")
        print("=" * 50)
        print(f"📋 Request: {request}")
        print("=" * 50)
        
        # Start the conversation between agents
        try:
            chat_result = self.user_proxy.initiate_chat(
                self.assistant,
                message=request,
                max_turns=10
            )
            return chat_result
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            return None

def main():
    """Main function to run the Stock Analysis System."""
    print("🎯 Multi-Agent Stock Analysis System")
    print("=" * 40)
    
    try:
        # Create the system
        system = StockAnalysisSystem()
        
        print("\nExample requests:")
        print("- 'Plot META and TESLA stock price YTD'")
        print("- 'Compare Apple and Microsoft performance'")
        print("- 'Show Amazon stock with moving averages'")
        print()
        
        # For testing - use a default request if no input available
        test_request = "Plot META and TESLA stock price change YTD"
        print(f"🧪 Running test with: {test_request}")
        system.start_analysis(test_request)
        
    except KeyboardInterrupt:
        print("\n👋 Analysis interrupted by user")
    except Exception as e:
        print(f"❌ System error: {e}")

if __name__ == "__main__":
    main()