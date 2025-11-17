import random
from fastmcp import FastMCP

mcp = FastMCP(name="")

if __name__ == "__main__":
    mcp.run(transport="http", host="localhost", port=8002)
