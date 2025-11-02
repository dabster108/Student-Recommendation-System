"""
Custom Validators and Utility Functions
========================================
Additional validation helpers for data integrity.
"""

from typing import Any, Dict, List, Optional
import re


def validate_string_length(
    value: str, 
    min_length: int = 1, 
    max_length: int = 1000,
    field_name: str = "field"
) -> str:
    """
    Validate string length within bounds.
    
    Args:
        value: String to validate
        min_length: Minimum allowed length
        max_length: Maximum allowed length
        field_name: Name of field for error messages
    
    Returns:
        str: Validated and stripped string
    
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    
    stripped = value.strip()
    
    if len(stripped) < min_length:
        raise ValueError(
            f"{field_name} must be at least {min_length} characters long"
        )
    
    if len(stripped) > max_length:
        raise ValueError(
            f"{field_name} must be at most {max_length} characters long"
        )
    
    return stripped


def validate_list_items(
    items: List[str],
    min_items: int = 1,
    max_items: int = 100,
    min_item_length: int = 1,
    max_item_length: int = 100,
    allow_duplicates: bool = False,
    field_name: str = "items"
) -> List[str]:
    """
    Validate a list of string items.
    
    Args:
        items: List of strings to validate
        min_items: Minimum number of items required
        max_items: Maximum number of items allowed
        min_item_length: Minimum length for each item
        max_item_length: Maximum length for each item
        allow_duplicates: Whether to allow duplicate items
        field_name: Name of field for error messages
    
    Returns:
        List[str]: Validated and cleaned list
    
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(items, list):
        raise ValueError(f"{field_name} must be a list")
    
    # Remove empty items and strip whitespace
    cleaned = [item.strip() for item in items if isinstance(item, str) and item.strip()]
    
    if len(cleaned) < min_items:
        raise ValueError(
            f"{field_name} must contain at least {min_items} item(s)"
        )
    
    if len(cleaned) > max_items:
        raise ValueError(
            f"{field_name} must contain at most {max_items} item(s)"
        )
    
    # Validate each item
    for item in cleaned:
        if len(item) < min_item_length:
            raise ValueError(
                f"Each {field_name} must be at least {min_item_length} characters long"
            )
        if len(item) > max_item_length:
            raise ValueError(
                f"Each {field_name} must be at most {max_item_length} characters long"
            )
    
    # Check for duplicates if not allowed
    if not allow_duplicates:
        lower_items = [item.lower() for item in cleaned]
        if len(lower_items) != len(set(lower_items)):
            raise ValueError(f"Duplicate items are not allowed in {field_name}")
    
    return cleaned


def validate_username(username: str) -> str:
    """
    Validate username format.
    
    Rules:
    - 3-50 characters
    - Only alphanumeric, underscore, and hyphen
    - Not a reserved word
    
    Args:
        username: Username to validate
    
    Returns:
        str: Validated username (lowercase)
    
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(username, str):
        raise ValueError("Username must be a string")
    
    username = username.strip().lower()
    
    if len(username) < 3:
        raise ValueError("Username must be at least 3 characters long")
    
    if len(username) > 50:
        raise ValueError("Username must be at most 50 characters long")
    
    if not re.match(r'^[a-z0-9_-]+$', username):
        raise ValueError(
            "Username can only contain lowercase letters, numbers, underscores, and hyphens"
        )
    
    # Reserved usernames
    reserved = ['admin', 'root', 'system', 'null', 'undefined', 'test', 'user']
    if username in reserved:
        raise ValueError(f"Username '{username}' is reserved and cannot be used")
    
    return username


def validate_name(name: str) -> str:
    """
    Validate person name format.
    
    Rules:
    - 2-100 characters
    - Only letters, spaces, hyphens, and apostrophes
    - At least one letter
    
    Args:
        name: Name to validate
    
    Returns:
        str: Validated name
    
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(name, str):
        raise ValueError("Name must be a string")
    
    name = name.strip()
    
    if len(name) < 2:
        raise ValueError("Name must be at least 2 characters long")
    
    if len(name) > 100:
        raise ValueError("Name must be at most 100 characters long")
    
    if not re.match(r'^[a-zA-Z\s\'-]+$', name):
        raise ValueError(
            "Name can only contain letters, spaces, hyphens, and apostrophes"
        )
    
    # Ensure at least one letter
    if not re.search(r'[a-zA-Z]', name):
        raise ValueError("Name must contain at least one letter")
    
    return name


def validate_query(query: str) -> str:
    """
    Validate search query.
    
    Rules:
    - 3-500 characters
    - Contains at least one alphanumeric character
    - Not just whitespace or special characters
    
    Args:
        query: Query to validate
    
    Returns:
        str: Validated query
    
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(query, str):
        raise ValueError("Query must be a string")
    
    query = query.strip()
    
    if len(query) < 3:
        raise ValueError("Query must be at least 3 characters long")
    
    if len(query) > 500:
        raise ValueError("Query must be at most 500 characters long")
    
    # Must contain at least one alphanumeric character
    if not re.search(r'[a-zA-Z0-9]', query):
        raise ValueError("Query must contain at least one alphanumeric character")
    
    return query


def validate_id(id_value: Any, field_name: str = "id") -> int:
    """
    Validate ID value.
    
    Args:
        id_value: ID to validate
        field_name: Name of field for error messages
    
    Returns:
        int: Validated ID
    
    Raises:
        ValueError: If validation fails
    """
    try:
        id_int = int(id_value)
    except (ValueError, TypeError):
        raise ValueError(f"{field_name} must be a valid integer")
    
    if id_int < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    
    return id_int


def sanitize_dict(
    data: Dict[str, Any],
    allowed_keys: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Sanitize dictionary by removing None values and optionally filtering keys.
    
    Args:
        data: Dictionary to sanitize
        allowed_keys: Optional list of allowed keys (None = allow all)
    
    Returns:
        Dict[str, Any]: Sanitized dictionary
    """
    if not isinstance(data, dict):
        return {}
    
    # Remove None values
    sanitized = {k: v for k, v in data.items() if v is not None}
    
    # Filter by allowed keys if provided
    if allowed_keys:
        sanitized = {k: v for k, v in sanitized.items() if k in allowed_keys}
    
    return sanitized


def validate_score(
    score: Any,
    min_score: float = 0.0,
    max_score: float = 1.0,
    field_name: str = "score"
) -> float:
    """
    Validate numeric score within bounds.
    
    Args:
        score: Score to validate
        min_score: Minimum allowed score
        max_score: Maximum allowed score
        field_name: Name of field for error messages
    
    Returns:
        float: Validated score
    
    Raises:
        ValueError: If validation fails
    """
    try:
        score_float = float(score)
    except (ValueError, TypeError):
        raise ValueError(f"{field_name} must be a valid number")
    
    if score_float < min_score or score_float > max_score:
        raise ValueError(
            f"{field_name} must be between {min_score} and {max_score}"
        )
    
    return score_float


def is_valid_json_data(data: Any) -> bool:
    """
    Check if data is valid for JSON serialization.
    
    Args:
        data: Data to check
    
    Returns:
        bool: True if valid, False otherwise
    """
    import json
    
    try:
        json.dumps(data)
        return True
    except (TypeError, ValueError):
        return False
