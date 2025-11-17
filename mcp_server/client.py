from fastmcp import Client
from google import genai
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

# Connect to local MCP server
mcp_client = Client("http://localhost:8002/mcp")

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

async def main():
    async with mcp_client:
        response = await gemini_client.aio.models.generate_content(
            model="gemini-2.0-flash",
            contents="Roll 3 dice!",
            config=genai.types.GenerateContentConfig(
                temperature=0,
                tools=[mcp_client.session],
            ),
        )
        print(response.text)

if __name__ == "__main__":
    asyncio.run(main())
