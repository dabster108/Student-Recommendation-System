# Student Recommendation System

A comprehensive AI-powered recommendation system for students built with FastAPI, featuring personalized content recommendations using TF-IDF, semantic analysis, and MCP (Model Context Protocol) integration with Google Gemini.

## 🌟 Features

- **Personalized Recommendations**: Content-based recommendations using TF-IDF and cosine similarity
- **Semantic Understanding**: Enhanced keyword extraction using Groq API
- **Multi-Category Support**: Forums, courses, wellness content, events, scholarships, and more
- **MCP Integration**: Model Context Protocol server for AI assistant integration
- **Comprehensive Validation**: Pydantic-based data validation with detailed error messages
- **RESTful API**: Well-documented FastAPI endpoints with OpenAPI/Swagger docs
- **Interactive UI**: Clean web interface for testing recommendations
- **Docker Support**: Containerized deployment with Docker and Docker Compose

## 📁 Project Structure

```
student_recommendation/
├── app/                          # Main application directory
│   ├── __init__.py
│   ├── main.py                   # FastAPI application & endpoints
│   ├── models.py                 # Pydantic data models & validation
│   ├── validators.py             # Custom validation functions
│   ├── data/
│   │   ├── dummy_data.json      # Sample recommendation data
│   │   └── student.json         # Student profiles storage
│   └── routes/
│       ├── __init__.py
│       └── recommendation.py     # Recommendation engine logic
│
├── mcp_server/                   # MCP Server for AI integration
│   ├── server.py                 # FastMCP server implementation
│   ├── client.py                 # MCP client with Gemini integration
│   ├── run_client.sh            # Client startup script
│   ├── pyproject.toml           # MCP dependencies
│   ├── .env                     # MCP environment variables
│   └── .python-version          # Python version specification
│
├── static/
│   └── index.html               # Web UI for testing
│
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker container configuration
├── docker-compose.yml           # Docker Compose setup
├── test_validation.py           # Validation tests
├── .env.example                 # Environment variables template
├── .gitignore
├── LICENSE
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker (optional, for containerized deployment)
- Groq API Key (for enhanced keyword extraction)
- Google Gemini API Key (for MCP client)

### Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/dabster108/Student-Recommendation-System.git
   cd student_recommendation
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys:
   # GROQ_API_KEY=your_groq_api_key_here
   ```

5. **Run the FastAPI server**
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

6. **Access the application**
   - Web UI: http://localhost:8000
   - API Docs: http://localhost:8000/api/docs
   - ReDoc: http://localhost:8000/api/redoc

### Docker Deployment

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

2. **Access the application**
   - Application: http://localhost:8000
   - API Documentation: http://localhost:8000/api/docs

## 🤖 MCP Server & AI Assistant

The project includes a Model Context Protocol (MCP) server that enables AI assistants like Google Gemini to interact with the recommendation system.

### What is MCP?

MCP (Model Context Protocol) is a standardized way for AI assistants to interact with external tools and APIs. Unlike traditional FastAPI + AI integration where the AI makes raw HTTP requests, MCP provides:

- **Standardized Interface**: AI tools speak MCP natively - no custom integration needed
- **Better Context Management**: MCP maintains conversation context across multiple tool calls
- **Simplified Tool Registration**: FastAPI endpoints automatically become AI-callable tools
- **Built-in Error Handling**: Automatic retry logic and error formatting

### Architecture

```
┌─────────────────┐         ┌──────────────┐         ┌─────────────────┐
│  Gemini Client  │ ◄─────► │  MCP Server  │ ◄─────► │  FastAPI App    │
│  (AI Assistant) │   MCP   │  (Tool Layer)│  HTTP   │  (Port: 8000)   │
└─────────────────┘         └──────────────┘         └─────────────────┘
```

**Flow**:
1. User asks Gemini: "Find me machine learning courses"
2. Gemini calls MCP tool: `recommend_learning(query="machine learning")`
3. MCP server translates to HTTP: `POST http://localhost:8000/recommend/learning`
4. FastAPI processes and returns recommendations
5. Gemini formats results in natural language

### Setting up MCP Server

1. **Ensure FastAPI server is running**
   ```bash
   # In terminal 1
   uvicorn app.main:app --reload --port 8000
   ```

2. **Navigate to MCP directory**
   ```bash
   # In terminal 2
   cd mcp_server
   ```

3. **Set up environment variables**
   ```bash
   # Create .env file in mcp_server/
   echo "GEMINI_API_KEY=your_gemini_api_key_here" > .env
   ```

4. **Install MCP dependencies**
   ```bash
   # Install uv if not already installed
   pip install uv
   
   # Install dependencies
   uv pip install -e .
   ```

5. **Run the MCP client**
   ```bash
   chmod +x run_client.sh
   ./run_client.sh
   
   # Or directly:
   uv run client.py
   ```

### MCP Client Modes

The MCP client offers two interaction modes:

#### 1. Demo Mode
Tests all recommendation endpoints automatically:
```bash
Choose mode:
1. Demo Mode - Test all recommendation endpoints
2. Chat Mode - Interactive AI assistant

Enter choice (1 or 2): 1

Running Demo Mode...
Testing: Get forum recommendations
✓ Success: Found 5 forums related to programming

Testing: Get learning recommendations
✓ Success: Found 3 courses on machine learning
...
```

#### 2. Chat Mode
Interactive AI assistant powered by Google Gemini:
```bash
Enter choice (1 or 2): 2

Chat Mode - Talk to Gemini AI
Type 'quit' to exit

You: Find me machine learning courses for beginners
AI: I found 3 excellent machine learning courses for you:
    1. Introduction to ML - Perfect for beginners, covers basics...
    2. Python for Data Science - Foundation course...
    3. Deep Learning Fundamentals - Next step after basics...

You: What tech events are happening this week?
AI: Here are the upcoming tech events:
    - AI Summit 2025 - Nov 25th at Convention Center...
    - Startup Hackathon - Nov 28-29th...
```

### MCP Features

- **Automatic Tool Discovery**: MCP server exposes all FastAPI POST endpoints as tools
- **Context-Aware Responses**: Gemini maintains conversation context and student preferences
- **Filtered Operations**: Only POST endpoints exposed (GET endpoints filtered out)
- **Error Handling**: Graceful error handling with user-friendly messages
- **Personalized Results**: Uses student profile data for better recommendations

## 📚 API Endpoints

### Student Management

- `POST /onboard` - Register a new student with interests and courses
- `GET /student` - Get current student profile
- `GET /student/{student_id}` - Get specific student by ID
- `GET /students` - Get all student profiles

**Example Request**:
```json
POST /onboard
{
  "username": "john_doe",
  "name": "John Doe",
  "email": "john@example.com",
  "faculty": "CSIT",
  "grade": "Grade 11",
  "interests": ["AI", "Web Development"],
  "courses": ["Python", "React"]
}
```

### Recommendation Endpoints

All recommendation endpoints support POST method with query parameters:

- `POST /recommend/forums` - Forum recommendations
- `POST /recommend/learning` - Learning content (courses, sets)
- `POST /recommend/wellness` - Wellness resources (challenges, activities)
- `POST /recommend/events` - Event recommendations
- `POST /recommend/opportunities` - Career opportunities
- `POST /recommend/scholarships` - Scholarship recommendations
- `POST /recommend/flashcards` - Flashcard sets
- `POST /recommend/qna` - Q&A content
- `POST /recommend/mcq` - MCQ practice sets
- `POST /recommend/truefalse` - True/False questions
- `POST /recommend/confessions` - Anonymous confessions

**Query Parameters**:
- `query` (optional): Search query text
- `student_id` (optional): Student ID for personalized recommendations

**Example Request**:
```bash
POST /recommend/learning?query=machine%20learning&student_id=1
```

**Example Response**:
```json
{
  "query": "machine learning",
  "student_id": 1,
  "total_results": 5,
  "recommendations": [
    {
      "item": {
        "id": 101,
        "title": "Introduction to Machine Learning",
        "description": "Comprehensive ML course for beginners",
        "category": "AI/ML",
        "tags": ["machine learning", "python", "ai"]
      },
      "similarity_score": 0.92,
      "rank": 1
    }
  ]
}
```

### Data Endpoints

- `GET /forums` - Get all forums
- `GET /learning` - Get all learning content
- `GET /wellness` - Get all wellness content
- `GET /events` - Get all events
- `GET /opportunities` - Get all opportunities
- `GET /scholarships` - Get all scholarships

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# Groq API Key for keyword extraction (optional but recommended)
GROQ_API_KEY=your_groq_api_key_here

# Optional: Custom port (default: 8000)
PORT=8000

# Optional: Environment
ENVIRONMENT=development
```

For MCP server, create `mcp_server/.env`:

```env
# Google Gemini API Key (required for MCP chat mode)
GEMINI_API_KEY=your_gemini_api_key_here
```

### Data Models

The system uses Pydantic models for validation:

- **StudentOnboard**: Student registration data with interests and courses
- **StudentProfile**: Student profile response model
- **RecommendationQuery**: Recommendation request with query and student_id
- **RecommendationResponse**: Recommendation results with scores and rankings
- **RecommendationItem**: Individual recommendation item with similarity score

See `app/models.py` for complete model definitions.

## 🔍 Recommendation Algorithm

The system uses a sophisticated multi-stage recommendation approach:

### 1. Query Enhancement
Enriches user query with student context:
```python
enhanced_query = f"{query} {student_interests} {student_courses} {student_faculty}"
```

### 2. TF-IDF Vectorization
Converts text content to numerical vectors:
- Analyzes term frequency across documents
- Considers document importance
- Creates feature vectors for similarity comparison

### 3. Cosine Similarity
Calculates similarity between query and content:
```
similarity = cosine(query_vector, content_vector)
```

### 4. Semantic Check (Optional)
Uses Groq API for deeper semantic understanding:
- Extracts key concepts from query
- Validates relevance with AI
- Filters out non-relevant results

### 5. Ranking & Personalization
- Sorts results by similarity scores
- Boosts content matching student profile
- Returns top N recommendations

See `app/routes/recommendation.py` for implementation details.

## 📊 Data Structure

Sample data is stored in `app/data/dummy_data.json`:

```json
{
  "forums": [
    {
      "id": 1,
      "title": "Python Programming Help",
      "description": "Get help with Python coding",
      "category": "Programming",
      "tags": ["python", "coding", "help"]
    }
  ],
  "learning_courses": [...],
  "wellness_challenges": [...],
  "events": [...],
  "scholarships": [...],
  "opportunities": [...]
}
```

Student profiles stored in `app/data/student.json`:

```json
{
  "students": [
    {
      "id": 1,
      "username": "john_doe",
      "name": "John Doe",
      "email": "john@example.com",
      "faculty": "CSIT",
      "grade": "Grade 11",
      "interests": ["AI", "Web Development"],
      "courses": ["Python", "React"]
    }
  ]
}
```

## 🧪 Testing

Run validation tests:

```bash
python test_validation.py
```

**Tests include**:
- ✓ Valid student data creation
- ✓ Invalid input handling
- ✓ Reserved username validation
- ✓ Faculty/grade enum validation
- ✓ Interest and course validation
- ✓ Query validation
- ✓ Email format validation

**Expected Output**:
```
Testing valid student data...
✓ Valid student created successfully

Testing invalid inputs...
✓ Caught validation error for invalid email
✓ Caught validation error for reserved username
...
All tests passed!
```

## 🐳 Docker Configuration

### Dockerfile

Production-ready container with:
- Python 3.11 slim base image
- Non-root user for security
- Health checks every 30s
- Optimized layer caching
- Exposed port 8000

### Docker Compose

Run the entire stack:

```bash
# Build and start
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

**Features**:
- Automatic container restarts
- Volume mounting for persistent data
- Health checks
- Network isolation
- Environment variable management

## 🛠️ Development

### Adding New Recommendation Types

1. **Add data to `app/data/dummy_data.json`**
   ```json
   "new_category": [
     {
       "id": 1,
       "title": "Sample Item",
       "description": "Description here",
       "tags": ["tag1", "tag2"]
     }
   ]
   ```

2. **Create recommendation function in `app/routes/recommendation.py`**
   ```python
   async def recommend_new_category(query: str, student_id: Optional[int] = None):
       return await generate_recommendations(
           query=query,
           content_items=load_dummy_data()["new_category"],
           student_id=student_id,
           content_type="new_category"
       )
   ```

3. **Add endpoint in `app/main.py`**
   ```python
   @app.post("/recommend/new-category")
   async def get_new_category_recommendations(
       query: str = "",
       student_id: Optional[int] = None
   ):
       return await recommend_new_category(query, student_id)
   ```

4. **MCP server automatically includes new endpoint** (no changes needed!)

### Code Style

- Follow PEP 8 guidelines
- Use type hints for all functions
- Document functions with docstrings
- Keep functions focused and modular

### Project Dependencies

**FastAPI Application** (`requirements.txt`):
- fastapi - Web framework
- uvicorn - ASGI server
- pydantic - Data validation
- scikit-learn - ML algorithms (TF-IDF, cosine similarity)
- groq - API client for semantic analysis
- python-dotenv - Environment variable management

**MCP Server** (`mcp_server/pyproject.toml`):
- fastmcp - MCP protocol implementation
- httpx - HTTP client for API calls
- google-generativeai - Gemini AI integration

## 📖 Use Cases

### For Students
- Discover relevant forums based on interests
- Find courses matching learning goals
- Get personalized event recommendations
- Explore scholarship opportunities
- Access wellness resources

### For Developers
- RESTful API for integration
- Extensible recommendation engine
- MCP server for AI assistant integration
- Docker deployment for production
- Well-documented codebase

### For AI Assistants
- Natural language interface via MCP
- Context-aware recommendations
- Multi-turn conversations
- Automatic tool discovery

## 🔐 Security

- Environment variables for API keys
- Non-root Docker user
- Input validation with Pydantic
- CORS configuration
- Health check endpoints

## 📈 Performance

- Efficient TF-IDF vectorization
- Caching of loaded data
- Async/await for concurrent requests
- Docker containerization for scalability

## 🗺️ Roadmap

- [ ] Add user authentication & authorization
- [ ] Implement collaborative filtering
- [ ] Add Redis caching layer
- [ ] Create admin dashboard
- [ ] Add real-time notifications
- [ ] Implement feedback loop for recommendations
- [ ] Add multi-language support
- [ ] Create mobile API endpoints

## 📝 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Support

For issues and questions:
- Open an issue on GitHub
- Check existing documentation
- Review API docs at `/api/docs`

## 🔗 Useful Links

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Groq API](https://console.groq.com/)
- [Google Gemini](https://ai.google.dev/)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [Docker Documentation](https://docs.docker.com/)

## 👥 Authors

- **dabster108** - *Initial work* - [GitHub](https://github.com/dabster108)

## 🙏 Acknowledgments

- FastAPI team for the excellent web framework
- Groq for semantic analysis capabilities
- Google for Gemini AI integration
- MCP community for protocol standards

---

**Version**: 1.0.0  
**Last Updated**: November 2025  
**Status**: Active Development

---

