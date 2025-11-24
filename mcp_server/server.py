import httpx
from fastmcp import FastMCP

# Create an HTTP client for your API
client = httpx.AsyncClient(base_url="http://127.0.0.1:8000")

#fastapi open ai spec

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

