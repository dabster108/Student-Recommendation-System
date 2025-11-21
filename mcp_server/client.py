import asyncio
import os
from dotenv import load_dotenv
from google import genai
# from fastmcp.client import Client
import httpx
from fastmcp import FastMCP, Client

# Load environment variables
load_dotenv()

# Create an HTTP client for your API
client = httpx.AsyncClient(base_url="http://127.0.0.1:8000")

# Load your OpenAPI spec 
openapi_spec = httpx.get("http://127.0.0.1:8000/openapi.json").json()

# Filter out GET recommendation endpoints - keep only POST
def filter_operations(spec):
    """Remove GET /recommend/* endpoints, keep only POST"""
    filtered_paths = {}
    for path, methods in spec.get("paths", {}).items():
        if path.startswith("/recommend/"):
            # Only keep POST methods for recommendation endpoints
            filtered_methods = {method: details for method, details in methods.items() if method == "post"}
            if filtered_methods:
                filtered_paths[path] = filtered_methods
        else:
            # Keep all other endpoints as-is
            filtered_paths[path] = methods
    
    spec["paths"] = filtered_paths
    return spec

# Filter the spec before creating MCP server
filtered_spec = filter_operations(openapi_spec)

# Create the MCP server with filtered spec
mcp = FastMCP.from_openapi(
    openapi_spec=filtered_spec,
    client=client,
    name="My API Server"
)
mcp_client = Client(mcp)
# MCP Server URL - MCP server runs on port 8001 (FastAPI app runs on 8000)
MCP_SERVER_URL = "http://localhost:8001/mcp"

# Initialize Gemini client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)



async def chat_loop():
    """Interactive chat loop with Gemini using MCP tools"""
    async with mcp_client as client:
        print("\n" + "=" * 60)
        print("Student Recommendation AI Assistant")
        print("=" * 60)
        print("Ask me for recommendations on:")
        print("  - Forums, Learning Materials, Wellness Resources")
        print("  - Events, Scholarships, Opportunities")
        print("  - Study Materials (Flashcards, Q&A, MCQs, True/False)")
        print("  - Confessions (Anonymous discussions)")
        print("\nType 'exit', 'quit', or 'q' to end the conversation")
        print("=" * 60 + "\n")
        
        # Chat history for context
        chat_history = []
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                # Check for exit commands
                if user_input.lower() in ['exit', 'quit', 'q', 'bye']:
                    print("\nThanks for using the Student Recommendation System!")
                    break
                
                if not user_input:
                    continue
                
                # Add user message to history
                chat_history.append({
                    "role": "user",
                    "parts": [{"text": user_input}]
                })
                
                
            
                print("\nAI is thinking...", end="", flush=True)
                
                # Generate response with Gemini using MCP tools
                response = await gemini_client.aio.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=chat_history,
                    config=genai.types.GenerateContentConfig(
                        temperature=0.7,
                        tools=[client.session],  # Pass MCP session as tool,
                    ),
                )
                
                print("\r" + " " * 30 + "\r", end="")  # Clear the "thinking" message
                
                # Extract and display response
                if response.text:
                    print(f"AI: {response.text}\n")
                    
                    # Add AI response to history
                    chat_history.append({
                        "role": "model",
                        "parts": [{"text": response.text}]
                    })
                else:
                    print("AI: I couldn't generate a response. Please try again.\n")
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n[ERROR] {e}\n")
                continue


async def main():
    """Main entry point"""
    print("\n" + "=" * 60)
    print("Student Recommendation MCP Client")
    print("=" * 60)
    print("\nChoose mode:")
    print("1. Demo Mode - Test all recommendation endpoints")
    print("2. Chat Mode - Interactive AI assistant")
    print("=" * 60)
  
  #ONLY RUN CHATLOOP
    await chat_loop()
    


if __name__ == "__main__":
    asyncio.run(main())
