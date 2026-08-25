import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from tactical_tools import get_acoustic_zones

# Load API keys
load_dotenv()

def build_tactical_advisor():
    """Initializes the AcousticGuard ReAct Agent using Groq."""
    
    # 1. Initialize the Groq LLM Engine (Lightning Fast)
    llm = ChatGroq(
        model="llama-3.3-70b-versatile", 
        temperature=0,
        max_retries=2
    )
    
    # 2. Give the LLM access to our custom physics tools
    tools = [get_acoustic_zones]
    
    # 3. Define the System Prompt (Prevents hallucination)
    system_prompt = """
    You are the AcousticGuard-OS Tactical Advisor. You are a highly advanced AI 
    assisting naval commanders and oceanographers. 
    
    RULES:
    1. NEVER guess about ocean depth, sound speed, or shadow zones.
    2. ALWAYS use the `get_acoustic_zones` tool when asked about specific locations.
    3. If the tool returns data about thermoclines, advise the commander that 
       submarines can hide beneath them.
    4. Keep your answers brief, tactical, and highly professional.
    """
    
    # 4. Compile the Agent
    # 4. Compile the Agent
    # 4. Compile the Agent
    agent = create_react_agent(llm, tools, prompt=system_prompt)
    return agent

if __name__ == "__main__":
    print("[*] Initializing AcousticGuard Tactical Advisor (Powered by Groq)...")
    advisor = build_tactical_advisor()
    
    print("[+] System Online. Enter a query (or type 'exit').")
    
    while True:
        user_input = input("\nCommander: ")
        if user_input.lower() == 'exit':
            break
            
        # Run the agent
        response = advisor.invoke({"messages": [("user", user_input)]})
        print(f"\nAdvisor: {response['messages'][-1].content}")