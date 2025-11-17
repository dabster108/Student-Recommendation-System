from fastapi import FastAPI
from fastmcp import FastMCP
import uvicorn

# Create FastAPI app for MCP tools/AI integration
app = FastAPI(
    title="Student Recommendation MCP Bridge",
    description="MCP Bridge for AI tool integration with student recommendations",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {"message": "Student Recommendation MCP Bridge Server", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "purpose": "AI tool integration"}

@app.get("/mcp")
async def mcp_info():
    return {
        "message": "MCP Server is running",
        "sse_endpoint": "http://127.0.0.1:8001/sse",
        "transport": "sse",
        "note": "Use /sse endpoint for MCP connections"
    }

# Mount FastAPI app to FastMCP for AI tool integration
mcp = FastMCP.from_fastapi(app)

if __name__ == "__main__":
    # Use FastMCP for AI tools, not HTTP proxy
    mcp.run(
        transport="sse",
        host="127.0.0.1",
        port=8001
    )
