# MCP Integration with FastAPI: Student Recommendation System

## 📋 Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [How MCP Works with FastAPI](#how-mcp-works-with-fastapi)
- [Working Mechanism](#working-mechanism)
- [Key Features](#key-features)
- [Components Breakdown](#components-breakdown)
- [Data Flow](#data-flow)
- [Integration Details](#integration-details)
- [API Endpoints](#api-endpoints)
- [MCP Server Implementation](#mcp-server-implementation)
- [Use Cases](#use-cases)
- [Benefits](#benefits)

---

## 🎯 Overview

This project demonstrates a **Student Recommendation System** that combines **FastAPI** (web framework), **FastMCP** (Model Context Protocol), and **AI models** (Groq, Google Gemini) to provide intelligent, personalized recommendations to students.

### What is MCP?

**MCP (Model Context Protocol)** is a protocol that enables AI models to interact with external tools and APIs in a standardized way. It acts as a bridge between Large Language Models (LLMs) and application APIs, allowing AI assistants to:
- Discover available API endpoints
- Understand API schemas
- Execute API calls dynamically
- Provide context-aware responses

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                          │
│                  (Web UI / Chat Interface)                      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MCP Client Layer                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Google Gemini AI Assistant                              │  │
│  │  - Natural language understanding                        │  │
│  │  - Tool selection & parameter extraction                 │  │
│  │  - Response generation                                   │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MCP Server Layer                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  FastMCP Server (server.py)                              │  │
│  │  - Exposes FastAPI endpoints as MCP tools                │  │
│  │  - Filters & transforms OpenAPI spec                     │  │
│  │  - Handles tool invocations                              │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Application                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Main Application (main.py)                              │  │
│  │  - REST API endpoints                                    │  │
│  │  - Data validation (Pydantic)                            │  │
│  │  - Error handling                                        │  │
│  │                                                           │  │
│  │  Recommendation Engine (recommendation.py)               │  │
│  │  - TF-IDF vectorization                                  │  │
│  │  - Cosine similarity matching                            │  │
│  │  - Groq API for semantic analysis                        │  │
│  │                                                           │  │
│  │  Data Models (models.py)                                 │  │
│  │  - Student profiles                                      │  │
│  │  - Recommendation queries                                │  │
│  │  - Response schemas                                      │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Data Layer                                 │
│  - student.json (Student profiles)                             │
│  - dummy_data.json (Recommendation content)                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

### Core Framework
- **FastAPI**: Modern, high-performance web framework
  - Automatic OpenAPI/Swagger documentation
  - Type validation with Pydantic
  - Async support
  - CORS middleware

### MCP Integration
- **FastMCP**: Python implementation of Model Context Protocol
  - Converts OpenAPI specs to MCP tools
  - Handles HTTP client communication
  - Provides tool discovery and invocation

### AI/ML Components
- **Groq API**: Fast LLM inference for keyword extraction
- **Google Gemini**: Conversational AI for MCP client
- **scikit-learn**: Machine learning library
  - TF-IDF vectorization
  - Cosine similarity computation

### Data Validation
- **Pydantic v2**: Advanced data validation and serialization
  - Custom validators
  - Field-level validation
  - Enum-based constraints

---

## 🔄 How MCP Works with FastAPI

### 1. **OpenAPI Specification Generation**
FastAPI automatically generates an OpenAPI 3.0 specification at `/openapi.json`:

```python
# FastAPI automatically exposes this endpoint
GET http://localhost:8000/openapi.json
```

The OpenAPI spec includes:
- All API endpoints (paths)
- HTTP methods (GET, POST, etc.)
- Request/response schemas
- Parameter definitions
- Authentication requirements

### 2. **FastMCP Server Creation**
The MCP server reads the OpenAPI spec and creates MCP tools:

```python
# mcp_server/server.py
import httpx
from fastmcp import FastMCP

# Create HTTP client for FastAPI
client = httpx.AsyncClient(base_url="http://127.0.0.1:8000")

# Fetch OpenAPI specification
openapi_spec = httpx.get("http://127.0.0.1:8000/openapi.json").json()

# Filter spec to include only desired endpoints
filtered_spec = filter_operations(openapi_spec)

# Create MCP server from OpenAPI spec
mcp = FastMCP.from_openapi(
    openapi_spec=filtered_spec,
    client=client,
    name="Student Recommendation API"
)
```

### 3. **Endpoint Filtering**
The system filters endpoints to expose only POST methods for recommendations:

```python
def filter_operations(spec):
    """Remove GET /recommend/* endpoints, keep only POST"""
    filtered_paths = {}
    for path, methods in spec.get("paths", {}).items():
        if path.startswith("/recommend/"):
            # Only keep POST methods
            filtered_methods = {
                method: details 
                for method, details in methods.items() 
                if method == "post"
            }
            if filtered_methods:
                filtered_paths[path] = filtered_methods
        else:
            filtered_paths[path] = methods
    
    spec["paths"] = filtered_paths
    return spec
```

### 4. **MCP Client Integration**
The MCP client uses Google Gemini to interact with the tools:

```python
# mcp_server/client.py
from fastmcp import FastMCP, Client
from google import genai

# Create MCP client
mcp_client = Client(mcp)

# Initialize Gemini
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

async with mcp_client as client:
    # AI can now use MCP tools to call FastAPI endpoints
    tools = await client.list_tools()
    
    # Gemini generates function calls based on user queries
    response = await gemini_client.aio.models.generate_content(
        model='gemini-2.0-flash-exp',
        contents=[...],
        tools=tools
    )
```

---

## ⚙️ Working Mechanism

### Step-by-Step Execution Flow

#### **1. User Interaction**
```
User: "I need practice materials for data structures"
```

#### **2. MCP Client Processing**
- User query is sent to Google Gemini
- Gemini analyzes the query
- Identifies relevant MCP tool: `POST /recommend/practice-sets`
- Extracts parameters from natural language

#### **3. Tool Invocation**
```json
{
  "tool_name": "post_recommend_practice_sets",
  "parameters": {
    "query": "data structures practice",
    "student_id": "student123",
    "top_k": 5
  }
}
```

#### **4. MCP Server Execution**
- FastMCP receives tool call
- Constructs HTTP request to FastAPI
- Sends POST request to `/recommend/practice-sets`

#### **5. FastAPI Processing**

**a. Request Validation**
```python
# models.py - Pydantic validation
class RecommendationQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    student_id: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=20)
```

**b. Student Profile Loading**
```python
# Load student data from student.json
student_data = load_student_profile(student_id)
```

**c. Query Enhancement**
```python
# recommendation.py
enhanced_query = enhance_query_with_student_context(
    query="data structures practice",
    student_data={
        "interests": ["algorithms", "coding"],
        "courses_enrolled": ["CS101", "CS201"]
    }
)
# Result: "data structures practice algorithms coding CS101 CS201"
```

**d. TF-IDF Vectorization**
```python
# Create TF-IDF vectors for query and all content
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform([enhanced_query] + content_texts)
```

**e. Cosine Similarity Calculation**
```python
# Compute similarity scores
similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
```

**f. Semantic Filtering (via Groq)**
```python
# Use Groq API to validate relevance
is_relevant, score = semantic_relevance_check(
    query=enhanced_query,
    item_text=content_description,
    threshold=0.30
)
```

**g. Ranking & Filtering**
```python
# Sort by similarity score
top_recommendations = sorted_items[:top_k]
```

#### **6. Response Generation**
```json
{
  "query": "data structures practice",
  "recommendations": [
    {
      "id": "practice_123",
      "title": "Data Structures Practice Set",
      "description": "Comprehensive practice questions...",
      "type": "practice_set",
      "similarity_score": 0.89,
      "tags": ["algorithms", "trees", "graphs"]
    }
  ],
  "total_found": 12,
  "student_context_used": true
}
```

#### **7. MCP Response to Client**
- FastMCP receives HTTP response
- Converts to MCP tool result
- Returns to Gemini

#### **8. AI Response to User**
```
Assistant: "I found excellent practice materials for data structures! 
Here's a comprehensive practice set covering algorithms, trees, and 
graphs that matches your interests. It includes 50+ coding problems 
with solutions. Would you like me to find more specific topics?"
```

---

## ✨ Key Features

### 1. **Multi-Category Recommendations**
The system provides recommendations across multiple content types:

| Category | Endpoint | Description |
|----------|----------|-------------|
| Forums | `/recommend/forums` | Discussion forums and communities |
| Learning Materials | `/recommend/learning-materials` | Courses, tutorials, bundles |
| Wellness | `/recommend/wellness` | Mental health, fitness resources |
| Events | `/recommend/events` | Campus events, workshops |
| Scholarships | `/recommend/scholarships` | Financial opportunities |
| Opportunities | `/recommend/opportunities` | Internships, jobs, competitions |
| Flashcards | `/recommend/flashcards` | Study flashcards |
| Q&A | `/recommend/qa` | Question-answer sets |
| MCQs | `/recommend/mcqs` | Multiple choice questions |
| True/False | `/recommend/true-false` | True/false questions |
| Practice Sets | `/recommend/practice-sets` | Comprehensive practice materials |
| Confessions | `/recommend/confessions` | Anonymous discussions |

### 2. **Personalization Engine**

**Student Profile Attributes:**
```python
class StudentOnboard(BaseModel):
    student_id: str
    email: EmailStr
    full_name: str
    faculty: Faculty  # Enum: Engineering, Science, etc.
    grade: Grade      # Enum: Freshman, Sophomore, etc.
    interests: List[str]  # AI, Web Dev, etc.
    courses_enrolled: List[str]
    preferred_language: str = "en"
```

**Context Enhancement:**
- Combines user query with student interests
- Incorporates enrolled courses
- Considers academic level
- Personalizes relevance scoring

### 3. **Hybrid Recommendation Algorithm**

**TF-IDF (Term Frequency-Inverse Document Frequency):**
- Statistical measure of word importance
- Identifies key terms in query and content
- Creates numerical vector representations

**Cosine Similarity:**
- Measures angle between vectors
- Range: 0 (unrelated) to 1 (identical)
- Finds most similar content

**Semantic Validation (Groq API):**
- LLM-based relevance checking
- Filters false positives
- Provides confidence scores

### 4. **Comprehensive Validation**

**Field-Level Validation:**
```python
class RecommendationQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    student_id: Optional[str] = Field(None, pattern=r'^[a-zA-Z0-9_-]+$')
    top_k: int = Field(default=5, ge=1, le=20)
    
    @field_validator('query')
    def validate_query(cls, v):
        if not v or v.isspace():
            raise ValueError("Query cannot be empty or whitespace")
        return v.strip()
```

**Error Handling:**
```python
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    errors = []
    for error in exc.errors():
        errors.append({
            "field": " -> ".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })
    return JSONResponse(
        status_code=422,
        content={"detail": "Validation error", "errors": errors}
    )
```

---

## 🧩 Components Breakdown

### **1. FastAPI Application (`app/main.py`)**

**Responsibilities:**
- Define REST API endpoints
- Handle HTTP requests/responses
- Implement CORS middleware
- Serve static files (web UI)
- Generate OpenAPI documentation

**Key Endpoints:**
```python
@app.post("/onboard", response_model=OnboardingResponse)
async def onboard_student(student: StudentOnboard)
    """Register new student profile"""

@app.post("/recommend/{category}", response_model=RecommendationResponse)
async def get_recommendations(category: str, query: RecommendationQuery)
    """Get personalized recommendations"""

@app.get("/health", response_model=HealthCheckResponse)
async def health_check()
    """System health status"""
```

### **2. Recommendation Engine (`app/routes/recommendation.py`)**

**Core Functions:**

```python
def enhance_query_with_student_context(query, student_data)
    """Enhance query with student interests and courses"""

def semantic_relevance_check(query, item_text, threshold)
    """Groq API-based relevance validation"""

def get_recommendations(query, student_id, content_data, top_k, content_type)
    """Main recommendation pipeline"""
    # 1. Load student profile
    # 2. Enhance query with context
    # 3. TF-IDF vectorization
    # 4. Cosine similarity calculation
    # 5. Semantic filtering
    # 6. Ranking and selection
    # 7. Return top K results
```

### **3. Data Models (`app/models.py`)**

**Pydantic Models:**
```python
class StudentOnboard(BaseModel)
    """Student registration data"""

class StudentProfile(BaseModel)
    """Complete student profile"""

class RecommendationQuery(BaseModel)
    """Recommendation request parameters"""

class RecommendationResponse(BaseModel)
    """Recommendation API response"""

class ContentType(str, Enum)
    """Supported content categories"""

class Faculty(str, Enum)
    """Academic faculties"""

class Grade(str, Enum)
    """Academic levels"""
```

### **4. MCP Server (`mcp_server/server.py`)**

**Responsibilities:**
- Load FastAPI OpenAPI specification
- Filter endpoints (only POST recommendations)
- Create FastMCP server instance
- Expose API as MCP tools

**Key Code:**
```python
# Load OpenAPI spec
openapi_spec = httpx.get("http://127.0.0.1:8000/openapi.json").json()

# Filter operations
filtered_spec = filter_operations(openapi_spec)

# Create MCP server
mcp = FastMCP.from_openapi(
    openapi_spec=filtered_spec,
    client=client,
    name="Student Recommendation API"
)
```

### **5. MCP Client (`mcp_server/client.py`)**

**Responsibilities:**
- Connect to MCP server
- Initialize Google Gemini AI
- Handle user chat interface
- Execute tool calls via Gemini
- Format and display responses

**Chat Loop:**
```python
async def chat_loop():
    async with mcp_client as client:
        # List available tools
        tools = await client.list_tools()
        
        while True:
            user_input = input("You: ")
            
            # Send to Gemini with tools
            response = await gemini_client.generate_content(
                model='gemini-2.0-flash-exp',
                contents=chat_history,
                tools=tools,
                tool_config={'function_calling_config': 'AUTO'}
            )
            
            # Execute tool calls
            for part in response.parts:
                if function_call := part.function_call:
                    result = await client.call_tool(
                        function_call.name,
                        function_call.args
                    )
```

---

## 📊 Data Flow

### Complete Request-Response Cycle

```
┌──────────┐
│   User   │
└────┬─────┘
     │ "Find me AI courses"
     ▼
┌──────────────────────┐
│  MCP Client          │
│  (Google Gemini)     │
│  - Parse intent      │
│  - Extract params    │
└────┬─────────────────┘
     │ Tool: post_recommend_learning_materials
     │ Args: {query: "AI courses", top_k: 5}
     ▼
┌──────────────────────┐
│  MCP Server          │
│  (FastMCP)           │
│  - Route to API      │
└────┬─────────────────┘
     │ POST /recommend/learning-materials
     │ Body: {query: "AI courses", top_k: 5}
     ▼
┌──────────────────────────────────┐
│  FastAPI Application              │
│  1. Validate request (Pydantic)   │
│  2. Load student profile          │
│  3. Enhance query context         │
└────┬─────────────────────────────┘
     │ Enhanced query: "AI courses machine learning python data science"
     ▼
┌──────────────────────────────────┐
│  Recommendation Engine            │
│  1. TF-IDF vectorization          │
│  2. Compute similarities          │
│  3. Semantic check (Groq)         │
│  4. Rank and filter               │
└────┬─────────────────────────────┘
     │ Top 5 results with scores
     ▼
┌──────────────────────┐
│  FastAPI Response    │
│  {recommendations:   │
│   [...],             │
│   total: 12}         │
└────┬─────────────────┘
     │ HTTP 200 + JSON
     ▼
┌──────────────────────┐
│  MCP Server          │
│  - Convert to        │
│    tool result       │
└────┬─────────────────┘
     │ MCP tool result
     ▼
┌──────────────────────┐
│  MCP Client          │
│  (Gemini)            │
│  - Generate response │
└────┬─────────────────┘
     │ Natural language response
     ▼
┌──────────┐
│   User   │ "Here are 5 AI courses perfect for you..."
└──────────┘
```

---

## 🔗 Integration Details

### **OpenAPI to MCP Tool Mapping**

FastMCP automatically converts OpenAPI operations to MCP tools:

**OpenAPI Endpoint:**
```yaml
paths:
  /recommend/forums:
    post:
      summary: Get forum recommendations
      operationId: recommend_forums
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/RecommendationQuery'
      responses:
        200:
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/RecommendationResponse'
```

**Becomes MCP Tool:**
```json
{
  "name": "post_recommend_forums",
  "description": "Get forum recommendations",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {"type": "string"},
      "student_id": {"type": "string"},
      "top_k": {"type": "integer"}
    },
    "required": ["query"]
  }
}
```

### **HTTP Client Configuration**

```python
# Async HTTP client with base URL
client = httpx.AsyncClient(
    base_url="http://127.0.0.1:8000",
    timeout=30.0,
    follow_redirects=True
)

# FastMCP uses this client for all API calls
mcp = FastMCP.from_openapi(
    openapi_spec=filtered_spec,
    client=client
)
```

### **Tool Discovery**

When MCP client connects, it discovers available tools:

```python
async with mcp_client as client:
    # Get all available tools
    tools = await client.list_tools()
    
    # Tools are passed to Gemini
    for tool in tools:
        print(f"Tool: {tool.name}")
        print(f"Description: {tool.description}")
        print(f"Parameters: {tool.inputSchema}")
```

---

## 🌐 API Endpoints

### **Health Check**
```http
GET /health
```
Returns system status and API information.

### **Student Onboarding**
```http
POST /onboard
Content-Type: application/json

{
  "student_id": "student123",
  "email": "student@example.com",
  "full_name": "John Doe",
  "faculty": "Engineering",
  "grade": "Junior",
  "interests": ["AI", "Web Development"],
  "courses_enrolled": ["CS101", "CS201"]
}
```

### **Recommendations**
All recommendation endpoints follow the same pattern:

```http
POST /recommend/{category}
Content-Type: application/json

{
  "query": "search query",
  "student_id": "student123",
  "top_k": 5
}
```

**Categories:**
- `forums` - Discussion forums
- `learning-materials` - Courses and bundles
- `wellness` - Wellness content
- `events` - Campus events
- `scholarships` - Financial aid
- `opportunities` - Internships, jobs
- `flashcards` - Study flashcards
- `qa` - Q&A sets
- `mcqs` - Multiple choice questions
- `true-false` - True/false questions
- `practice-sets` - Practice materials
- `confessions` - Anonymous discussions

---

## 🖥️ MCP Server Implementation

### **Starting the MCP Server**

The MCP server exposes FastAPI endpoints as tools that AI can use:

```python
# mcp_server/server.py

import httpx
from fastmcp import FastMCP

# 1. Create HTTP client
client = httpx.AsyncClient(base_url="http://127.0.0.1:8000")

# 2. Fetch OpenAPI spec
openapi_spec = httpx.get("http://127.0.0.1:8000/openapi.json").json()

# 3. Filter endpoints (optional)
def filter_operations(spec):
    filtered_paths = {}
    for path, methods in spec.get("paths", {}).items():
        if path.startswith("/recommend/"):
            # Only POST methods for recommendations
            filtered_methods = {
                k: v for k, v in methods.items() if k == "post"
            }
            if filtered_methods:
                filtered_paths[path] = filtered_methods
        else:
            filtered_paths[path] = methods
    spec["paths"] = filtered_paths
    return spec

filtered_spec = filter_operations(openapi_spec)

# 4. Create MCP server
mcp = FastMCP.from_openapi(
    openapi_spec=filtered_spec,
    client=client,
    name="Student Recommendation API"
)

# Server is now ready to accept tool calls
```

### **MCP Client with Gemini**

```python
# mcp_server/client.py

import asyncio
from google import genai
from fastmcp import Client

# Initialize clients
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
mcp_client = Client(mcp)

async def chat_loop():
    async with mcp_client as client:
        # Get available tools
        tools = await client.list_tools()
        
        # Chat history for context
        chat_history = []
        
        while True:
            user_input = input("You: ")
            
            # Add to history
            chat_history.append({
                "role": "user",
                "parts": [{"text": user_input}]
            })
            
            # Get response from Gemini
            response = await gemini_client.aio.models.generate_content(
                model='gemini-2.0-flash-exp',
                contents=chat_history,
                tools=tools,
                tool_config={
                    'function_calling_config': 'AUTO'
                }
            )
            
            # Handle tool calls
            for part in response.parts:
                if function_call := part.function_call:
                    # Execute tool via MCP
                    result = await client.call_tool(
                        function_call.name,
                        function_call.args
                    )
                    
                    # Add result to history
                    chat_history.append({
                        "role": "function",
                        "parts": [{
                            "function_response": {
                                "name": function_call.name,
                                "response": result
                            }
                        }]
                    })
            
            # Get final response
            final_response = await gemini_client.aio.models.generate_content(
                model='gemini-2.0-flash-exp',
                contents=chat_history
            )
            
            print(f"Assistant: {final_response.text}")
```

---

## 💡 Use Cases

### **1. Academic Support**
```
User: "I'm struggling with calculus. Can you find study materials?"

System Flow:
- Gemini identifies need for learning materials
- Calls: post_recommend_learning_materials(query="calculus help")
- Returns: Calculus courses, tutorials, video lectures
- Response: "Here are 5 comprehensive calculus resources..."
```

### **2. Career Guidance**
```
User: "Show me internship opportunities in software engineering"

System Flow:
- Tool: post_recommend_opportunities(query="software engineering internship")
- Considers student's CS major and coding interests
- Returns: Relevant internships with descriptions
```

### **3. Exam Preparation**
```
User: "I need practice questions for data structures exam"

System Flow:
- Tools called:
  - post_recommend_mcqs(query="data structures")
  - post_recommend_practice_sets(query="data structures")
- Returns: MCQs, practice problems, flashcards
```

### **4. Wellness Support**
```
User: "Feeling stressed about finals. Any wellness resources?"

System Flow:
- Tool: post_recommend_wellness(query="stress management finals")
- Returns: Meditation apps, counseling services, stress relief activities
```

### **5. Community Engagement**
```
User: "Are there any AI forums I can join?"

System Flow:
- Tool: post_recommend_forums(query="artificial intelligence")
- Returns: AI discussion forums, student groups
```

---

## 🎁 Benefits

### **1. Natural Language Interface**
- Users ask questions in plain English
- No need to learn API syntax
- Conversational experience

### **2. Intelligent Tool Selection**
- AI automatically selects appropriate API endpoints
- Handles multiple tool calls if needed
- Chains operations logically

### **3. Context-Aware Recommendations**
- Remembers student profile
- Considers previous queries
- Provides personalized results

### **4. Scalable Architecture**
- FastAPI handles high concurrency
- Async operations throughout
- Horizontal scaling with Docker

### **5. Type Safety**
- Pydantic validation ensures data integrity
- Prevents invalid API calls
- Clear error messages

### **6. Extensibility**
- Easy to add new endpoints
- Automatically exposed as MCP tools
- No client-side updates needed

### **7. Developer Experience**
- Auto-generated API documentation
- Clear separation of concerns
- Well-structured codebase

---

## 🚀 Running the System

### **1. Start FastAPI Server**
```bash
cd /Users/dikshanta/Documents/MCP-FASTMCP/student_recommendation
uvicorn app.main:app --reload --port 8000
```

### **2. Verify API is Running**
```bash
# Check health
curl http://localhost:8000/health

# View OpenAPI spec
curl http://localhost:8000/openapi.json
```

### **3. Start MCP Client**
```bash
cd mcp_server
python client.py
```

### **4. Interact with AI**
```
You: I need study materials for Python programming