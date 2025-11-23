"""
Data Models and Validation
===========================
Pydantic models for comprehensive data validation with type hints.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Dict, Any, Optional, Union, Literal
from enum import Enum
import re


# ============================================
# Enums for Validation
# ============================================

class Faculty(str, Enum):
    """Valid faculty options"""
    ENGINEERING = "Engineering"
    SCIENCE = "Science"
    BUSINESS = "Business"
    ARTS = "Arts"
    MEDICINE = "Medicine"
    LAW = "Law"
    EDUCATION = "Education"
    OTHER = "Other"


class Grade(str, Enum):
    """Valid grade/year options"""
    FRESHMAN = "Freshman"
    SOPHOMORE = "Sophomore"
    JUNIOR = "Junior"
    SENIOR = "Senior"
    GRADUATE = "Graduate"


class ContentType(str, Enum):
    """Types of content in the system"""
    FORUM = "forum"
    BUNDLE = "bundle"
    COURSE = "course"
    LEARNING_SET = "learning_set"
    PRACTICE_SET = "practice_set"
    CHALLENGE = "challenge"
    WELLNESS_CONTENT = "wellness_content"
    ACTIVITY = "activity"
    OPPORTUNITY = "opportunity"
    EVENT = "event"
    SCHOLARSHIP = "scholarship"
    CONFESSION = "confession"
    FLASHCARD = "flashcard"
    QNA = "qna"
    TRUE_FALSE = "true_false"
    MCQ = "mcq"


# ============================================
# Request Models (Input Validation)
# ============================================

class StudentOnboard(BaseModel):
    """
    Model for student onboarding data with comprehensive validation.
    """
    name: str = Field(
        ..., 
        min_length=2, 
        max_length=100,
        description="Student's full name"
    )
    username: str = Field(
        ..., 
        min_length=3, 
        max_length=50,
        pattern=r'^[a-zA-Z0-9_-]+$',
        description="Username (alphanumeric, underscore, hyphen only)"
    )
    address: str = Field(
        ..., 
        min_length=5, 
        max_length=200,
        description="Student's address"
    )
    faculty: Faculty = Field(
        ...,
        description="Student's faculty/department"
    )
    grade: Grade = Field(
        ...,
        description="Student's current grade/year"
    )
    interests: List[str] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="List of student interests (1-20 items)"
    )
    courses_enrolled: List[str] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="List of enrolled courses (1-20 items)"
    )

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name contains only letters and spaces"""
        if not v.strip():
            raise ValueError("Name cannot be empty or just whitespace")
        if not re.match(r'^[a-zA-Z\s\'-]+$', v):
            raise ValueError("Name can only contain letters, spaces, hyphens, and apostrophes")
        return v.strip()

    @field_validator('username')
    @classmethod
    def validate_username(cls, v: str) -> str:
        """Validate username format"""
        if not v.strip():
            raise ValueError("Username cannot be empty or just whitespace")
        if v.lower() in ['admin', 'root', 'system', 'null', 'undefined']:
            raise ValueError("Username is reserved and cannot be used")
        return v.strip().lower()

    @field_validator('address')
    @classmethod
    def validate_address(cls, v: str) -> str:
        """Validate address is not empty"""
        if not v.strip():
            raise ValueError("Address cannot be empty or just whitespace")
        return v.strip()

    @field_validator('interests')
    @classmethod
    def validate_interests(cls, v: List[str]) -> List[str]:
        """Validate interests list"""
        if not v:
            raise ValueError("At least one interest is required")
        
        # Remove empty strings and strip whitespace
        cleaned = [interest.strip() for interest in v if interest.strip()]
        
        if not cleaned:
            raise ValueError("At least one non-empty interest is required")
        
        # Check for duplicates (case-insensitive)
        lower_interests = [i.lower() for i in cleaned]
        if len(lower_interests) != len(set(lower_interests)):
            raise ValueError("Duplicate interests are not allowed")
        
        # Validate each interest
        for interest in cleaned:
            if len(interest) < 2:
                raise ValueError(f"Interest '{interest}' is too short (minimum 2 characters)")
            if len(interest) > 50:
                raise ValueError(f"Interest '{interest}' is too long (maximum 50 characters)")
        
        return cleaned

    @field_validator('courses_enrolled')
    @classmethod
    def validate_courses(cls, v: List[str]) -> List[str]:
        """Validate courses list"""
        if not v:
            raise ValueError("At least one course is required")
        
        # Remove empty strings and strip whitespace
        cleaned = [course.strip() for course in v if course.strip()]
        
        if not cleaned:
            raise ValueError("At least one non-empty course is required")
        
        # Check for duplicates (case-insensitive)
        lower_courses = [c.lower() for c in cleaned]
        if len(lower_courses) != len(set(lower_courses)):
            raise ValueError("Duplicate courses are not allowed")
        
        # Validate each course
        for course in cleaned:
            if len(course) < 2:
                raise ValueError(f"Course '{course}' is too short (minimum 2 characters)")
            if len(course) > 100:
                raise ValueError(f"Course '{course}' is too long (maximum 100 characters)")
        
        return cleaned

    class Config:
        json_schema_extra = {
            "example": {
                "name": "John Doe",
                "username": "johndoe123",
                "address": "123 University Ave, City, State",
                "faculty": "Engineering",
                "grade": "Junior",
                "interests": ["Machine Learning", "Web Development", "Data Science"],
                "courses_enrolled": ["CS101", "MATH201", "PHYS101"]
            }
        }


class RecommendationQuery(BaseModel):
    """
    Model for recommendation query with validation.
    """
    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Search query for recommendations"
    )
    student_id: Optional[int] = Field(
        None,
        ge=1,
        description="Optional student ID for personalized recommendations"
    )

    @field_validator('query')
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate query is not empty and contains meaningful content"""
        if not v.strip():
            raise ValueError("Query cannot be empty or just whitespace")
        
        # Check if query contains at least one alphanumeric character
        if not re.search(r'[a-zA-Z0-9]', v):
            raise ValueError("Query must contain at least one alphanumeric character")
        
        return v.strip()

    class Config:
        json_schema_extra = {
            "example": {
                "query": "I want to learn about machine learning",
                "student_id": 1
            }
        }


# ============================================
# Response Models (Output Validation)
# ============================================

class StudentProfile(BaseModel):
    """
    Model for student profile data in responses.
    """
    id: int = Field(..., ge=1, description="Student ID")
    name: str = Field(..., min_length=2, max_length=100)
    username: str = Field(..., min_length=3, max_length=50)
    address: str = Field(..., min_length=5, max_length=200)
    faculty: str
    grade: str
    interests: List[str] = Field(..., min_length=1)
    courses_enrolled: List[str] = Field(..., min_length=1)


class RecommendationItem(BaseModel):
    """
    Model for a single recommendation item with scores.
    """
    id: str = Field(..., description="Item ID")
    title: str = Field(..., min_length=1, max_length=500)
    description: Optional[str] = Field(None, max_length=5000)
    content_type: Optional[str] = None
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    tfidf_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    semantic_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Additional metadata fields (all optional)
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    member_count: Optional[int] = Field(None, ge=0)
    post_count: Optional[int] = Field(None, ge=0)
    card_count: Optional[int] = Field(None, ge=0)
    duration: Optional[str] = None
    difficulty: Optional[str] = None
    instructor: Optional[str] = None
    rating: Optional[float] = Field(None, ge=0.0, le=5.0)
    date: Optional[str] = None
    location: Optional[str] = None
    deadline: Optional[str] = None
    amount: Optional[str] = None
    eligibility: Optional[List[str]] = None
    organizer: Optional[str] = None
    question: Optional[str] = None
    answer: Optional[str] = None
    content: Optional[str] = None
    
    class Config:
        extra = "allow"  # Allow additional fields from dummy data


class RecommendationResponse(BaseModel):
    """
    Model for recommendation response with full validation.
    """
    student: Optional[StudentProfile] = Field(
        None,
        description="Student profile if available"
    )
    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Original query"
    )
    extracted_keywords: List[str] = Field(
        ...,
        description="Keywords extracted from query"
    )
    recommendations: List[Dict[str, Any]] = Field(
        ...,
        description="List of recommended items"
    )

    @field_validator('extracted_keywords')
    @classmethod
    def validate_keywords(cls, v: List[str]) -> List[str]:
        """Validate keywords list"""
        if not v:
            return []
        return [kw.strip() for kw in v if kw.strip()]

    @field_validator('recommendations')
    @classmethod
    def validate_recommendations(cls, v: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate recommendations list"""
        if v is None:
            return []
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "student": {
                    "id": 1,
                    "name": "John Doe",
                    "username": "johndoe123",
                    "address": "123 University Ave",
                    "faculty": "Engineering",
                    "grade": "Junior",
                    "interests": ["Machine Learning", "Web Development"],
                    "courses_enrolled": ["CS101", "MATH201"]
                },
                "query": "I want to learn about machine learning",
                "extracted_keywords": ["machine learning", "artificial intelligence", "data science"],
                "recommendations": [
                    {
                        "id": "course1",
                        "title": "Introduction to Machine Learning",
                        "similarity_score": 0.95
                    }
                ]
            }
        }


class OnboardingResponse(BaseModel):
    """
    Response model for student onboarding.
    """
    message: str
    student: StudentProfile


class ErrorResponse(BaseModel):
    """
    Standard error response model.
    """
    detail: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code for client handling")
    
    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Invalid input data",
                "error_code": "VALIDATION_ERROR"
            }
        }


class HealthCheckResponse(BaseModel):
    """
    API health check response.
    """
    message: str
    version: str
    status: Literal["running", "degraded", "error"]
    groq_enabled: bool
    data_loaded: bool = True
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Student Recommendation API",
                "version": "1.0.0",
                "status": "running",
                "groq_enabled": True,
                "data_loaded": True
            }
        }


# ============================================
# Internal Data Models
# ============================================

class StudentData(BaseModel):
    """
    Internal model for student data storage with all fields.
    """
    id: int = Field(..., ge=1)
    name: str = Field(..., min_length=2, max_length=100)
    username: str = Field(..., min_length=3, max_length=50)
    address: str = Field(..., min_length=5, max_length=200)
    faculty: str
    grade: str
    interests: List[str] = Field(..., min_length=1)
    courses_enrolled: List[str] = Field(..., min_length=1)
    
    class Config:
        extra = "allow"  # Allow additional fields for future extensions




class ValidationError(BaseModel):
    """
    Detailed validation error model.
    """
    field: str
    message: str
    type: str
    

class ValidationErrorResponse(BaseModel):
    """
    Response for validation errors with details.
    """
    detail: str = "Validation error"
    errors: List[ValidationError]
