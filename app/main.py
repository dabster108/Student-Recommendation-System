"""
Student Recommendation POC - FastAPI Backen
A prototype recommendation system that collects student info and provides 
personalized content recommendations using TF-IDF + Cosine Similarity with 
Groq API for keyword extraction.
cd /Users/dikshanta/Documents/Recommendation_Poc/student_recommendation && python3 -m uvicorn app.main:app --reload --port 8000
"""
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import json
import os
from dotenv import load_dotenv

# Machine Learning imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Groq API import
from groq import Groq

# Import recommendation engine
from .routes import recommendation as rec_engine

# Import data models
from .models import (
    StudentOnboard,
    RecommendationQuery,
    RecommendationResponse,
    OnboardingResponse,
    StudentProfile,
    HealthCheckResponse,
    ErrorResponse,
    ValidationErrorResponse,
    ValidationError as ValidationErrorModel
)

# Load environment variables from specific path
ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# Initialize FastAPI app
app = FastAPI(
    title="Student Recommendation API",
    description="AI-powered recommendation system for students with comprehensive data validation",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add CORS middleware to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# Exception Handlers for Validation
# ============================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """
    Custom handler for Pydantic validation errors.
    Returns detailed, user-friendly error messages.
    """
    errors = []
    for error in exc.errors():
        field_path = " -> ".join(str(loc) for loc in error["loc"])
        errors.append(
            ValidationErrorModel(
                field=field_path,
                message=error["msg"],
                type=error["type"]
            ).dict()
        )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Validation error occurred. Please check your input data.",
            "errors": errors
        }
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """
    Custom handler for ValueError exceptions.
    """
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "detail": str(exc),
            "error_code": "VALUE_ERROR"
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    General exception handler for unexpected errors.
    """
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An unexpected error occurred. Please try again later.",
            "error_code": "INTERNAL_SERVER_ERROR"
        }
    )

# File paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR.parent / "static"
DUMMY_DATA_PATH = DATA_DIR / "dummy_data.json"
STUDENT_DATA_PATH = DATA_DIR / "student.json"

# Mount static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Load dummy data with validation
def load_dummy_data() -> Dict[str, List[Dict[str, Any]]]:
    """
    Load and validate dummy data from JSON file.
    """
    try:
        with open(DUMMY_DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Validate data structure
        if not isinstance(data, dict):
            raise ValueError("Dummy data must be a dictionary")
        
        # Validate each category is a list
        for category, items in data.items():
            if not isinstance(items, list):
                raise ValueError(f"Category '{category}' must be a list")
        
        print(f"✓ Dummy data loaded successfully with {len(data)} categories")
        return data
    
    except FileNotFoundError:
        raise RuntimeError(f"Dummy data file not found at {DUMMY_DATA_PATH}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON in dummy data file: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading dummy data: {e}")


DUMMY_DATA = load_dummy_data()

# Initialize Groq client with validation
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("Warning: GROQ_API_KEY not found in environment variables")
    groq_client = None
else:
    print(f"✓ Groq API key loaded successfully (key starts with: {GROQ_API_KEY[:10]}...)")
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        print("✓ Groq client initialized successfully")
    except Exception as e:
        print(f"✗ Error initializing Groq client: {e}")
        groq_client = None


# ============================================
# Helper Functions with Type Hints
# ============================================

def load_all_students() -> List[Dict[str, Any]]:
    """Load all student profiles from student.json.

    Returns an empty list when the file doesn't exist or cannot be parsed.
    Accepts both legacy single-dict and list formats.
    """
    if not STUDENT_DATA_PATH.exists():
        return []

    try:
        with open(STUDENT_DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return [data]
            return []
    except (json.JSONDecodeError, Exception):
        return []


def get_next_student_id() -> int:
    """Determine the next incremental numeric student id."""
    students = load_all_students()
    if not students:
        return 1
    ids = [s.get("id", 0) for s in students if isinstance(s.get("id", None), int)]
    return (max(ids) + 1) if ids else 1


def load_student_data() -> Optional[Dict[str, Any]]:
    """Backward-compatible loader that returns a single student when available.

    If multiple students exist, returns the first one to preserve legacy behavior.
    Use `load_all_students()` or `get_student_by_id()` for multi-student handling.
    """
    students = load_all_students()
    if not students:
        return None
    return students[0]


def save_student_data(student_input: Union[Dict[str, Any], List[Dict[str, Any]]]) -> None:
    """Save student data.

    - If a dict is provided, append it to the stored list of students.
    - If a list is provided, overwrite the storage with that list.
    """
    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if isinstance(student_input, dict):
        students = load_all_students()
        students.append(student_input)
        to_write = students
    elif isinstance(student_input, list):
        to_write = student_input
    else:
        raise ValueError("student_input must be a dict or list of dicts")

    with open(STUDENT_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(to_write, f, indent=2, ensure_ascii=False)


def get_student_by_id(student_id: int) -> Optional[Dict[str, Any]]:
    """Return student dict matching the provided id, or None."""
    students = load_all_students()
    for s in students:
        try:
            if int(s.get("id", -1)) == int(student_id):
                return s
        except Exception:
            continue
    return None


def resolve_student_for_request(student_id: Optional[int]) -> Optional[Dict[str, Any]]:
    """Resolve which student profile to use for a recommendation request.

    Behavior:
    - If student_id is provided: return that student's profile (or None if not found).
    - If no student_id and exactly one student exists: return that student's profile (legacy support).
    - Otherwise return None (anonymous request).
    """
    if student_id is not None:
        return get_student_by_id(student_id)

    students = load_all_students()
    if len(students) == 1:
        return students[0]
    return None


def extract_keywords_with_groq(query: str) -> List[str]:
    """
    Use Groq API to extract important keywords/intent from user query.
    Falls back to simple keyword extraction if API is unavailable.
    """
    if not groq_client:
        print("Groq API not available, using fallback keyword extraction")
        return fallback_keyword_extraction(query)
    
    try:
        system_prompt = """You are an intelligent keyword extraction assistant.

Analyze the user's query and extract keywords for what they WANT.
- Pay attention to words like: "want", "need", "interested in", "looking for"
- IGNORE words after: "don't want", "not interested", "avoid", "except", "but not"
- Add related synonyms and similar topics for positive preferences
- Never include topics the user explicitly rejects

Examples:
Input: "I don't want yoga but I want workout"
Output: workout, fitness, gym, strength training, cardio

Input: "I need physics help but not chemistry"
Output: physics, mechanics, thermodynamics, quantum

Input: "looking for engineering forums not medicine"
Output: engineering, technology, coding, programming, robotics

Return only comma-separated keywords for what the user WANTS."""

        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"Extract keywords from: {query}"
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=100
        )
        
        keywords_text = chat_completion.choices[0].message.content.strip()
        keywords = [kw.strip().lower() for kw in keywords_text.split(",")]
        
        print(f"Groq extracted keywords: {keywords}")
        return keywords
        
    except Exception as e:
        print(f"Groq API error: {e}, using fallback")
        return fallback_keyword_extraction(query)


def fallback_keyword_extraction(query: str) -> List[str]:
    """
    Simple fallback keyword extraction using basic NLP techniques.
    
    Args:
        query: User search query
    
    Returns:
        List[str]: Extracted keywords (max 5)
    """
    stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 
                  'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 
                  'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
                  'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
                  'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am',
                  'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
                  'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
                  'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
                  'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
                  'through', 'during', 'before', 'after', 'above', 'below', 'to',
                  'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
                  'again', 'further', 'then', 'once', 'want', 'dont', 'not'}
    
    words = query.lower().split()
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    
    return keywords[:5] if keywords else [query.lower()]


def get_top_matches(
    query: str,
    dataset: List[Dict[str, Any]],
    key: str = "description",
    top_n: int = 5
) -> List[Dict[str, Any]]:
    """
    Content-based recommendation using TF-IDF + Cosine Similarity.
    
    Args:
        query: Search query
        dataset: List of items to search
        key: Key to use for text content
        top_n: Number of results to return
    
    Returns:
        List[Dict[str, Any]]: Top N matched items with similarity scores
    """
    if not dataset:
        return []
    
    texts = []
    valid_indices = []
    
    for idx, item in enumerate(dataset):
        text_content = item.get(key, "")
        
        if "title" in item and key != "title":
            text_content = f"{item.get('title', '')} {text_content}"
        
        if "tags" in item:
            tags = item.get("tags", [])
            if isinstance(tags, list):
                text_content = f"{text_content} {' '.join(tags)}"
        
        if text_content and text_content.strip():
            texts.append(text_content.strip())
            valid_indices.append(idx)
    
    if not texts:
        return []
    
    corpus = texts + [query]
    
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=1000,
        ngram_range=(1, 2)
    )
    
    tfidf_matrix = vectorizer.fit_transform(corpus)
    query_vector = tfidf_matrix[-1]
    doc_vectors = tfidf_matrix[:-1]
    similarities = cosine_similarity(query_vector, doc_vectors).flatten()
    top_indices = np.argsort(similarities)[::-1][:top_n]
    
    results = []
    for idx in top_indices:
        original_idx = valid_indices[idx]
        item_copy = dataset[original_idx].copy()
        item_copy["similarity_score"] = float(similarities[idx])
        results.append(item_copy)
    
    return results


def recommend_from_category(
    query: str,
    category: str,
    top_n: int = 5
) -> List[Dict[str, Any]]:
    """
    Get recommendations from a specific category in dummy data.
    
    Args:
        query: Search query
        category: Category name (e.g., 'forums', 'events')
        top_n: Number of results to return
    
    Returns:
        List[Dict[str, Any]]: Recommendations from the category
    
    Raises:
        HTTPException: If category doesn't exist
    """
    if category not in DUMMY_DATA:
        raise HTTPException(
            status_code=404,
            detail=f"Category '{category}' not found in dummy data"
        )
    
    dataset = DUMMY_DATA[category]
    
    if dataset and len(dataset) > 0:
        first_item = dataset[0]
        if "description" in first_item:
            key = "description"
        elif "content" in first_item:
            key = "content"
        elif "question" in first_item:
            key = "question"
        else:
            key = "title"
    else:
        key = "description"
    
    return get_top_matches(query, dataset, key=key, top_n=top_n)



@app.get("/", response_model=HealthCheckResponse)
async def home() -> Union[FileResponse, HealthCheckResponse]:
    """Serve the main HTML page or return API info"""
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    return HealthCheckResponse(
        message="Student Recommendation API",
        version="1.0.0",
        status="running",
        groq_enabled=groq_client is not None,
        data_loaded=bool(DUMMY_DATA)
    )

@app.get("/api", response_model=HealthCheckResponse)
async def api_info() -> HealthCheckResponse:
    """API health check endpoint with detailed status"""
    return HealthCheckResponse(
        message="Student Recommendation API",
        version="1.0.0",
        status="running",
        groq_enabled=groq_client is not None,
        data_loaded=bool(DUMMY_DATA)
    )


@app.post("/onboard", response_model=OnboardingResponse, status_code=status.HTTP_201_CREATED)
async def onboard_student(student: StudentOnboard) -> OnboardingResponse:
    """
    Onboard a new student by saving their profile information.
    
    Validates all input fields and automatically generates incremental ID.
    
    Args:
        student: StudentOnboard model with validated student data
    
    Returns:
        OnboardingResponse with success message and student profile
    
    Raises:
        HTTPException: If data validation fails or save operation fails
    """
    student_data = student.model_dump()
    student_data["id"] = get_next_student_id()
    
    try:
        save_student_data(student_data)
        
        # Create validated response
        student_profile = StudentProfile(**student_data)
        
        return OnboardingResponse(
            message="Student onboarded successfully",
            student=student_profile
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save student data: {str(e)}"
        )


@app.get("/student", response_model=StudentProfile)
async def get_student() -> StudentProfile:
    """
    Get current student profile (first student if multiple exist).
    
    Returns:
        StudentProfile: Validated student profile data
    
    Raises:
        HTTPException: If no student profile exists
    """
    student_data = load_student_data()
    
    if not student_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No student profile found. Please onboard first using POST /onboard"
        )
    
    return StudentProfile(**student_data)


@app.get("/student/{student_id}", response_model=StudentProfile)
async def get_student_by_id_endpoint(student_id: int) -> StudentProfile:
    """
    Get a specific student profile by ID.
    
    Args:
        student_id: The student ID to retrieve
    
    Returns:
        StudentProfile: Validated student profile data
    
    Raises:
        HTTPException: If student not found
    """
    student_data = get_student_by_id(student_id)
    
    if not student_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Student with ID {student_id} not found"
        )
    
    return StudentProfile(**student_data)


@app.get("/students", response_model=List[StudentProfile])
async def get_all_students() -> List[StudentProfile]:
    """
    Get all student profiles.
    
    Returns:
        List[StudentProfile]: List of all student profiles
    """
    students = load_all_students()
    return [StudentProfile(**student) for student in students]


@app.get("/recommend/forums", response_model=RecommendationResponse)
async def recommend_forums_get(query: str, student_id: Optional[int] = None) -> RecommendationResponse:
    """
    Recommend forums for a student.
    
    Args:
        query: Search query
        student_id: Optional student ID (query parameter)
    
    Returns:
        RecommendationResponse: Personalized forum recommendations
    """
    student_data = resolve_student_for_request(student_id)
    keywords = extract_keywords_with_groq(query)
    
    forums_data = DUMMY_DATA.get("forums", [])
    recommendations = rec_engine.recommend_forums(
        query=query,
        forums_data=forums_data,
        student_data=student_data,
        top_n=5
    )
    
    student_profile = StudentProfile(**student_data) if student_data else None
    
    return RecommendationResponse(
        student=student_profile,
        query=query,
        extracted_keywords=keywords,
        recommendations=recommendations
    )


@app.post("/recommend/forums", response_model=RecommendationResponse)
async def recommend_forums_post(query: str, student_id: Optional[int] = None) -> RecommendationResponse:
    """
    Recommend forums for a student (POST version).
    
    Args:
        query: Search query
        student_id: Optional student ID (query parameter)
    
    Returns:
        RecommendationResponse: Personalized forum recommendations
    """
    student_data = resolve_student_for_request(student_id)
    keywords = extract_keywords_with_groq(query)
    
    forums_data = DUMMY_DATA.get("forums", [])
    recommendations = rec_engine.recommend_forums(
        query=query,
        forums_data=forums_data,
        student_data=student_data,
        top_n=5
    )
    
    student_profile = StudentProfile(**student_data) if student_data else None
    
    return RecommendationResponse(
        student=student_profile,
        query=query,
        extracted_keywords=keywords,
        recommendations=recommendations
    )


@app.post("/recommend/learning", response_model=RecommendationResponse)
async def recommend_learning_post(query: str, student_id: Optional[int] = None) -> RecommendationResponse:
    """
    Recommend learning content for a student.
    
    Args:
        query: Search query
        student_id: Optional student ID (query parameter)
    
    Returns:
        RecommendationResponse: Personalized learning recommendations
    """
    student_data = resolve_student_for_request(student_id)
    keywords = extract_keywords_with_groq(query)
    
    recommendations = rec_engine.recommend_learning_content(
        query=query,
        bundles=DUMMY_DATA.get("bundles", []),
        courses=DUMMY_DATA.get("learning_courses", []),
        learning_sets=DUMMY_DATA.get("learning_sets", []),
        practice_sets=DUMMY_DATA.get("practice_sets", []),
        student_data=student_data,
        top_n=10
    )
    
    student_profile = StudentProfile(**student_data) if student_data else None
    
    return RecommendationResponse(
        student=student_profile,
        query=query,
        extracted_keywords=keywords,
        recommendations=recommendations
    )


@app.get("/recommend/learning", response_model=RecommendationResponse)
async def recommend_learning_get(query: str, student_id: Optional[int] = None) -> RecommendationResponse:
    """
    Recommend learning content for a student.
    
    Args:
        query: Search query
        student_id: Optional student ID (query parameter)
    
    Returns:
        RecommendationResponse: Personalized learning recommendations
    """
    student_data = resolve_student_for_request(student_id)
    keywords = extract_keywords_with_groq(query)
    
    recommendations = rec_engine.recommend_learning_content(
        query=query,
        bundles=DUMMY_DATA.get("bundles", []),
        courses=DUMMY_DATA.get("learning_courses", []),
        learning_sets=DUMMY_DATA.get("learning_sets", []),
        practice_sets=DUMMY_DATA.get("practice_sets", []),
        student_data=student_data,
        top_n=10
    )
    
    student_profile = StudentProfile(**student_data) if student_data else None
    
    return RecommendationResponse(
        student=student_profile,
        query=query,
        extracted_keywords=keywords,
        recommendations=recommendations
    )


@app.post("/recommend/wellness", response_model=RecommendationResponse)
async def recommend_wellness_post(query: str, student_id: Optional[int] = None) -> RecommendationResponse:
    """
    Recommend wellness content for a student.
    
    Args:
        query: Search query
        student_id: Optional student ID (query parameter)
    
    Returns:
        RecommendationResponse: Wellness content recommendations
    """
    student_data = resolve_student_for_request(student_id)
    keywords = extract_keywords_with_groq(query)
    
    recommendations = rec_engine.recommend_wellness(
        query=query,
        challenges=DUMMY_DATA.get("wellness_challenges", []),
        content=DUMMY_DATA.get("wellness_content", []),
        activities=DUMMY_DATA.get("wellness_activities", []),
        student_data=student_data,
        top_n=5
    )
    
    student_profile = StudentProfile(**student_data) if student_data else None
    
    return RecommendationResponse(
        student=student_profile,
        query=query,
        extracted_keywords=keywords,
        recommendations=recommendations
    )


@app.post("/recommend/opportunities", response_model=RecommendationResponse)
async def recommend_opportunities_post(query: str, student_id: Optional[int] = None) -> RecommendationResponse:
    """
    Recommend opportunities for a student.
    
    Args:
        query: Search query
        student_id: Optional student ID (query parameter)
    
    Returns:
        RecommendationResponse: Opportunities recommendations
    """
    student_data = resolve_student_for_request(student_id)
    keywords = extract_keywords_with_groq(query)
    
    opportunities_data = DUMMY_DATA.get("opportunities", [])
    recommendations = rec_engine.recommend_opportunities(
        query=query,
        opportunities_data=opportunities_data,
        student_data=student_data,
        top_n=5
    )
    
    student_profile = StudentProfile(**student_data) if student_data else None
    
    return RecommendationResponse(
        student=student_profile,
        query=query,
        extracted_keywords=keywords,
        recommendations=recommendations
    )


@app.post("/recommend/events", response_model=RecommendationResponse)
async def recommend_events_post(query: str, student_id: Optional[int] = None) -> RecommendationResponse:
    """
    Recommend events for a student.
    
    Args:
        query: Search query
        student_id: Optional student ID (query parameter)
    
    Returns:
        RecommendationResponse: Events recommendations
    """
    student_data = resolve_student_for_request(student_id)
    keywords = extract_keywords_with_groq(query)
    
    events_data = DUMMY_DATA.get("events", [])
    recommendations = rec_engine.recommend_events(
        query=query,
        events_data=events_data,
        student_data=student_data,
        top_n=5
    )
    
    student_profile = StudentProfile(**student_data) if student_data else None
    
    return RecommendationResponse(
        student=student_profile,
        query=query,
        extracted_keywords=keywords,
        recommendations=recommendations
    )


@app.post("/recommend/scholarships", response_model=RecommendationResponse)
async def recommend_scholarships_post(query: str, student_id: Optional[int] = None) -> RecommendationResponse:
    """
    Recommend scholarships for a student.
    
    Args:
        query: Search query
        student_id: Optional student ID (query parameter)
    
    Returns:
        RecommendationResponse: Scholarships recommendations
    """
    student_data = resolve_student_for_request(student_id)
    keywords = extract_keywords_with_groq(query)
    
    scholarships_data = DUMMY_DATA.get("scholarships", [])
    recommendations = rec_engine.recommend_scholarships(
        query=query,
        scholarships_data=scholarships_data,
        student_data=student_data,
        top_n=5
    )
    
    student_profile = StudentProfile(**student_data) if student_data else None
    
    return RecommendationResponse(
        student=student_profile,
        query=query,
        extracted_keywords=keywords,
        recommendations=recommendations
    )


@app.post("/recommend/confessions", response_model=RecommendationResponse)
async def recommend_confessions_post(query: str, student_id: Optional[int] = None) -> RecommendationResponse:
    """
    Recommend confessions for a student.
    
    Args:
        query: Search query
        student_id: Optional student ID (query parameter)
    
    Returns:
        RecommendationResponse: Confessions recommendations
    """
    student_data = resolve_student_for_request(student_id)
    keywords = extract_keywords_with_groq(query)
    
    confessions_data = DUMMY_DATA.get("confessions", [])
    recommendations = rec_engine.recommend_confessions(
        query=query,
        confessions_data=confessions_data,
        student_data=student_data,
        top_n=5
    )
    
    student_profile = StudentProfile(**student_data) if student_data else None
    
    return RecommendationResponse(
        student=student_profile,
        query=query,
        extracted_keywords=keywords,
        recommendations=recommendations
    )


@app.post("/recommend/flashcards", response_model=RecommendationResponse)
async def recommend_flashcards_post(query: str, student_id: Optional[int] = None) -> RecommendationResponse:
    """
    Recommend flashcards for a student.
    
    Args:
        query: Search query
        student_id: Optional student ID (query parameter)
    
    Returns:
        RecommendationResponse: Flashcards recommendations
    """
    student_data = resolve_student_for_request(student_id)
    keywords = extract_keywords_with_groq(query)
    
    flashcards_data = DUMMY_DATA.get("flashcards", [])
    recommendations = rec_engine.recommend_flashcards(
        query=query,
        flashcards_data=flashcards_data,
        student_data=student_data,
        top_n=5
    )
    
    student_profile = StudentProfile(**student_data) if student_data else None
    
    return RecommendationResponse(
        student=student_profile,
        query=query,
        extracted_keywords=keywords,
        recommendations=recommendations
    )


@app.post("/recommend/qna", response_model=RecommendationResponse)
async def recommend_qna_post(query: str, student_id: Optional[int] = None) -> RecommendationResponse:
    """
    Recommend Q&A content for a student.
    
    Args:
        query: Search query
        student_id: Optional student ID (query parameter)
    
    Returns:
        RecommendationResponse: Q&A recommendations
    """
    student_data = resolve_student_for_request(student_id)
    keywords = extract_keywords_with_groq(query)
    
    qna_data = DUMMY_DATA.get("qna", [])
    recommendations = rec_engine.recommend_qna(
        query=query,
        qna_data=qna_data,
        student_data=student_data,
        top_n=5
    )
    
    student_profile = StudentProfile(**student_data) if student_data else None
    
    return RecommendationResponse(
        student=student_profile,
        query=query,
        extracted_keywords=keywords,
        recommendations=recommendations
    )


@app.post("/recommend/truefalse", response_model=RecommendationResponse)
async def recommend_truefalse_post(query: str, student_id: Optional[int] = None) -> RecommendationResponse:
    """
    Recommend True/False question sets for a student.
    
    Args:
        query: Search query
        student_id: Optional student ID (query parameter)
    
    Returns:
        RecommendationResponse: True/False recommendations
    """
    student_data = resolve_student_for_request(student_id)
    keywords = extract_keywords_with_groq(query)
    
    truefalse_data = DUMMY_DATA.get("true_false", [])
    recommendations = rec_engine.recommend_truefalse(
        query=query,
        truefalse_data=truefalse_data,
        student_data=student_data,
        top_n=5
    )
    
    student_profile = StudentProfile(**student_data) if student_data else None
    
    return RecommendationResponse(
        student=student_profile,
        query=query,
        extracted_keywords=keywords,
        recommendations=recommendations
    )


@app.post("/recommend/mcq", response_model=RecommendationResponse)
async def recommend_mcq_post(query: str, student_id: Optional[int] = None) -> RecommendationResponse:
    """
    Recommend MCQ sets for a student.
    
    Args:
        query: Search query
        student_id: Optional student ID (query parameter)
    
    Returns:
        RecommendationResponse: MCQ recommendations
    """
    student_data = resolve_student_for_request(student_id)
    keywords = extract_keywords_with_groq(query)
    
    mcq_data = DUMMY_DATA.get("mcq", [])
    recommendations = rec_engine.recommend_mcq(
        query=query,
        mcq_data=mcq_data,
        student_data=student_data,
        top_n=5
    )
    
    student_profile = StudentProfile(**student_data) if student_data else None
    
    return RecommendationResponse(
        student=student_profile,
        query=query,
        extracted_keywords=keywords,
        recommendations=recommendations
    )


# ============================================
# Legacy GET endpoints with type hints
# ============================================

@app.get("/forums", response_model=List[Dict[str, Any]])
async def get_forums() -> List[Dict[str, Any]]:
    """Get all forums"""
    return DUMMY_DATA.get("forums", [])


@app.get("/learning", response_model=Dict[str, List[Dict[str, Any]]])
async def get_learning_content() -> Dict[str, List[Dict[str, Any]]]:
    """Get all learning content"""
    return {
        "bundles": DUMMY_DATA.get("bundles", []),
        "learning_courses": DUMMY_DATA.get("learning_courses", []),
        "learning_sets": DUMMY_DATA.get("learning_sets", []),
        "practice_sets": DUMMY_DATA.get("practice_sets", [])
    }


@app.get("/wellness", response_model=Dict[str, List[Dict[str, Any]]])
async def get_wellness() -> Dict[str, List[Dict[str, Any]]]:
    """Get all wellness content"""
    return {
        "wellness_challenges": DUMMY_DATA.get("wellness_challenges", []),
        "wellness_content": DUMMY_DATA.get("wellness_content", []),
        "wellness_activities": DUMMY_DATA.get("wellness_activities", [])
    }


@app.get("/opportunities", response_model=List[Dict[str, Any]])
async def get_opportunities() -> List[Dict[str, Any]]:
    """Get all opportunities"""
    return DUMMY_DATA.get("opportunities", [])


@app.get("/events", response_model=List[Dict[str, Any]])
async def get_events() -> List[Dict[str, Any]]:
    """Get all events"""
    return DUMMY_DATA.get("events", [])


@app.get("/scholarships", response_model=List[Dict[str, Any]])
async def get_scholarships() -> List[Dict[str, Any]]:
    """Get all scholarships"""
    return DUMMY_DATA.get("scholarships", [])


@app.get("/confessions", response_model=List[Dict[str, Any]])
async def get_confessions() -> List[Dict[str, Any]]:
    """Get all confessions"""
    return DUMMY_DATA.get("confessions", [])


@app.get("/flashcards", response_model=List[Dict[str, Any]])
async def get_flashcards() -> List[Dict[str, Any]]:
    """Get all flashcards"""
    return DUMMY_DATA.get("flashcards", [])


@app.get("/qna", response_model=List[Dict[str, Any]])
async def get_qna() -> List[Dict[str, Any]]:
    """Get all Q&A"""
    return DUMMY_DATA.get("qna", [])


@app.get("/truefalse", response_model=List[Dict[str, Any]])
async def get_truefalse() -> List[Dict[str, Any]]:
    """Get all True/False questions"""
    return DUMMY_DATA.get("true_false", [])


@app.get("/mcq", response_model=List[Dict[str, Any]])
async def get_mcq() -> List[Dict[str, Any]]:
    """Get all MCQ sets"""
    return DUMMY_DATA.get("mcq", [])


# ============================================
# Student-Specific Recommendation Endpoints (GET with Path Parameters)
# ============================================

@app.get("/recommend/wellness", response_model=RecommendationResponse)
async def recommend_wellness_get(query: str, student_id: Optional[int] = None) -> RecommendationResponse:
    """Recommend wellness content for a student."""
    student_data = resolve_student_for_request(student_id)
    keywords = extract_keywords_with_groq(query)
    
    recommendations = rec_engine.recommend_wellness(
        query=query,
        challenges=DUMMY_DATA.get("wellness_challenges", []),
        content=DUMMY_DATA.get("wellness_content", []),
        activities=DUMMY_DATA.get("wellness_activities", []),
        student_data=student_data,
        top_n=5
    )
    
    student_profile = StudentProfile(**student_data) if student_data else None
    
    return RecommendationResponse(
        student=student_profile,
        query=query,
        extracted_keywords=keywords,
        recommendations=recommendations
    )


@app.get("/recommend/opportunities", response_model=RecommendationResponse)
async def recommend_opportunities_get(query: str, student_id: Optional[int] = None) -> RecommendationResponse:
    """Recommend opportunities for a student."""
    student_data = resolve_student_for_request(student_id)
    keywords = extract_keywords_with_groq(query)
    
    opportunities_data = DUMMY_DATA.get("opportunities", [])
    recommendations = rec_engine.recommend_opportunities(
        query=query,
        opportunities_data=opportunities_data,
        student_data=student_data,
        top_n=5
    )
    
    student_profile = StudentProfile(**student_data) if student_data else None
    
    return RecommendationResponse(
        student=student_profile,
        query=query,
        extracted_keywords=keywords,
        recommendations=recommendations
    )


@app.get("/recommend/events", response_model=RecommendationResponse)
async def recommend_events_get(query: str, student_id: Optional[int] = None) -> RecommendationResponse:
    """Recommend events for a student."""
    student_data = resolve_student_for_request(student_id)
    keywords = extract_keywords_with_groq(query)
    
    events_data = DUMMY_DATA.get("events", [])
    recommendations = rec_engine.recommend_events(
        query=query,
        events_data=events_data,
        student_data=student_data,
        top_n=5
    )
    
    student_profile = StudentProfile(**student_data) if student_data else None
    
    return RecommendationResponse(
        student=student_profile,
        query=query,
        extracted_keywords=keywords,
        recommendations=recommendations
    )


@app.get("/recommend/scholarships", response_model=RecommendationResponse)
async def recommend_scholarships_get(query: str, student_id: Optional[int] = None) -> RecommendationResponse:
    """Recommend scholarships for a student."""
    student_data = resolve_student_for_request(student_id)
    keywords = extract_keywords_with_groq(query)
    
    scholarships_data = DUMMY_DATA.get("scholarships", [])
    recommendations = rec_engine.recommend_scholarships(
        query=query,
        scholarships_data=scholarships_data,
        student_data=student_data,
        top_n=5
    )
    
    student_profile = StudentProfile(**student_data) if student_data else None
    
    return RecommendationResponse(
        student=student_profile,
        query=query,
        extracted_keywords=keywords,
        recommendations=recommendations
    )


@app.get("/recommend/confessions", response_model=RecommendationResponse)
async def recommend_confessions_get(query: str, student_id: Optional[int] = None) -> RecommendationResponse:
    """Recommend confessions for a student."""
    student_data = resolve_student_for_request(student_id)
    keywords = extract_keywords_with_groq(query)
    
    confessions_data = DUMMY_DATA.get("confessions", [])
    recommendations = rec_engine.recommend_confessions(
        query=query,
        confessions_data=confessions_data,
        student_data=student_data,
        top_n=5
    )
    
    student_profile = StudentProfile(**student_data) if student_data else None
    
    return RecommendationResponse(
        student=student_profile,
        query=query,
        extracted_keywords=keywords,
        recommendations=recommendations
    )


@app.get("/recommend/flashcards", response_model=RecommendationResponse)
async def recommend_flashcards_get(query: str, student_id: Optional[int] = None) -> RecommendationResponse:
    """Recommend flashcards for a student."""
    student_data = resolve_student_for_request(student_id)
    keywords = extract_keywords_with_groq(query)
    
    flashcards_data = DUMMY_DATA.get("flashcards", [])
    recommendations = rec_engine.recommend_flashcards(
        query=query,
        flashcards_data=flashcards_data,
        student_data=student_data,
        top_n=5
    )
    
    student_profile = StudentProfile(**student_data) if student_data else None
    
    return RecommendationResponse(
        student=student_profile,
        query=query,
        extracted_keywords=keywords,
        recommendations=recommendations
    )


@app.get("/recommend/qna", response_model=RecommendationResponse)
async def recommend_qna_get(query: str, student_id: Optional[int] = None) -> RecommendationResponse:
    """Recommend Q&A for a student."""
    student_data = resolve_student_for_request(student_id)
    keywords = extract_keywords_with_groq(query)
    
    qna_data = DUMMY_DATA.get("qna", [])
    recommendations = rec_engine.recommend_qna(
        query=query,
        qna_data=qna_data,
        student_data=student_data,
        top_n=5
    )
    
    student_profile = StudentProfile(**student_data) if student_data else None
    
    return RecommendationResponse(
        student=student_profile,
        query=query,
        extracted_keywords=keywords,
        recommendations=recommendations
    )


@app.get("/recommend/truefalse", response_model=RecommendationResponse)
async def recommend_truefalse_get(query: str, student_id: Optional[int] = None) -> RecommendationResponse:
    """Recommend True/False questions for a student."""
    student_data = resolve_student_for_request(student_id)
    keywords = extract_keywords_with_groq(query)
    
    truefalse_data = DUMMY_DATA.get("true_false", [])
    recommendations = rec_engine.recommend_truefalse(
        query=query,
        truefalse_data=truefalse_data,
        student_data=student_data,
        top_n=5
    )
    
    student_profile = StudentProfile(**student_data) if student_data else None
    
    return RecommendationResponse(
        student=student_profile,
        query=query,
        extracted_keywords=keywords,
        recommendations=recommendations
    )


@app.get("/recommend/mcq", response_model=RecommendationResponse)
async def recommend_mcq_get(query: str, student_id: Optional[int] = None) -> RecommendationResponse:
    """Recommend MCQ sets for a student."""
    student_data = resolve_student_for_request(student_id)
    keywords = extract_keywords_with_groq(query)
    
    mcq_data = DUMMY_DATA.get("mcq", [])
    recommendations = rec_engine.recommend_mcq(
        query=query,
        mcq_data=mcq_data,
        student_data=student_data,
        top_n=5
    )
    
    student_profile = StudentProfile(**student_data) if student_data else None
    
    return RecommendationResponse(
        student=student_profile,
        query=query,
        extracted_keywords=keywords,
        recommendations=recommendations
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)