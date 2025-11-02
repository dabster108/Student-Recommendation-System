"""
Test Script for Data Validation
================================
Run this to verify that data validation is working correctly.
"""

import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pydantic import ValidationError
from app.models import StudentOnboard, RecommendationQuery


def test_valid_student():
    """Test valid student data."""
    print("✓ Testing valid student data...")
    try:
        student = StudentOnboard(
            name="John Doe",
            username="johndoe123",
            address="123 University Ave, City, State",
            faculty="Engineering",
            grade="Junior",
            interests=["Machine Learning", "Web Development"],
            courses_enrolled=["CS101", "MATH201"]
        )
        print(f"  ✓ Valid student created: {student.name}")
        return True
    except ValidationError as e:
        print(f"  ✗ Unexpected validation error: {e}")
        return False


def test_invalid_name():
    """Test invalid name (too short)."""
    print("\n✓ Testing invalid name (too short)...")
    try:
        student = StudentOnboard(
            name="J",  # Too short
            username="johndoe123",
            address="123 University Ave, City, State",
            faculty="Engineering",
            grade="Junior",
            interests=["Machine Learning"],
            courses_enrolled=["CS101"]
        )
        print("  ✗ Should have raised validation error")
        return False
    except ValidationError as e:
        print(f"  ✓ Correctly caught validation error: Name too short")
        return True


def test_invalid_username():
    """Test invalid username (reserved word)."""
    print("\n✓ Testing invalid username (reserved word)...")
    try:
        student = StudentOnboard(
            name="John Doe",
            username="admin",  # Reserved
            address="123 University Ave, City, State",
            faculty="Engineering",
            grade="Junior",
            interests=["Machine Learning"],
            courses_enrolled=["CS101"]
        )
        print("  ✗ Should have raised validation error")
        return False
    except ValidationError as e:
        print(f"  ✓ Correctly caught validation error: Reserved username")
        return True


def test_invalid_interests():
    """Test invalid interests (empty list)."""
    print("\n✓ Testing invalid interests (empty list)...")
    try:
        student = StudentOnboard(
            name="John Doe",
            username="johndoe123",
            address="123 University Ave, City, State",
            faculty="Engineering",
            grade="Junior",
            interests=[],  # Empty
            courses_enrolled=["CS101"]
        )
        print("  ✗ Should have raised validation error")
        return False
    except ValidationError as e:
        print(f"  ✓ Correctly caught validation error: Empty interests")
        return True


def test_duplicate_interests():
    """Test duplicate interests."""
    print("\n✓ Testing duplicate interests...")
    try:
        student = StudentOnboard(
            name="John Doe",
            username="johndoe123",
            address="123 University Ave, City, State",
            faculty="Engineering",
            grade="Junior",
            interests=["coding", "Coding", "CODING"],  # Duplicates
            courses_enrolled=["CS101"]
        )
        print("  ✗ Should have raised validation error")
        return False
    except ValidationError as e:
        print(f"  ✓ Correctly caught validation error: Duplicate interests")
        return True


def test_invalid_faculty():
    """Test invalid faculty enum."""
    print("\n✓ Testing invalid faculty...")
    try:
        student = StudentOnboard(
            name="John Doe",
            username="johndoe123",
            address="123 University Ave, City, State",
            faculty="InvalidFaculty",  # Not in enum
            grade="Junior",
            interests=["Machine Learning"],
            courses_enrolled=["CS101"]
        )
        print("  ✗ Should have raised validation error")
        return False
    except ValidationError as e:
        print(f"  ✓ Correctly caught validation error: Invalid faculty")
        return True


def test_valid_query():
    """Test valid recommendation query."""
    print("\n✓ Testing valid recommendation query...")
    try:
        query = RecommendationQuery(
            query="I want to learn machine learning",
            student_id=1
        )
        print(f"  ✓ Valid query created: {query.query}")
        return True
    except ValidationError as e:
        print(f"  ✗ Unexpected validation error: {e}")
        return False


def test_invalid_query_short():
    """Test invalid query (too short)."""
    print("\n✓ Testing invalid query (too short)...")
    try:
        query = RecommendationQuery(
            query="ML",  # Too short
            student_id=1
        )
        print("  ✗ Should have raised validation error")
        return False
    except ValidationError as e:
        print(f"  ✓ Correctly caught validation error: Query too short")
        return True


def test_invalid_query_empty():
    """Test invalid query (whitespace only)."""
    print("\n✓ Testing invalid query (whitespace only)...")
    try:
        query = RecommendationQuery(
            query="   ",  # Whitespace only
            student_id=1
        )
        print("  ✗ Should have raised validation error")
        return False
    except ValidationError as e:
        print(f"  ✓ Correctly caught validation error: Empty query")
        return True


def test_invalid_student_id():
    """Test invalid student ID (negative)."""
    print("\n✓ Testing invalid student ID (negative)...")
    try:
        query = RecommendationQuery(
            query="I want to learn machine learning",
            student_id=-1  # Negative
        )
        print("  ✗ Should have raised validation error")
        return False
    except ValidationError as e:
        print(f"  ✓ Correctly caught validation error: Negative student ID")
        return True


def main():
    """Run all validation tests."""
    print("="*60)
    print("Data Validation Test Suite")
    print("="*60)
    
    tests = [
        test_valid_student,
        test_invalid_name,
        test_invalid_username,
        test_invalid_interests,
        test_duplicate_interests,
        test_invalid_faculty,
        test_valid_query,
        test_invalid_query_short,
        test_invalid_query_empty,
        test_invalid_student_id
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "="*60)
    print(f"Test Results: {sum(results)}/{len(results)} passed")
    print("="*60)
    
    if all(results):
        print("\n🎉 All validation tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
