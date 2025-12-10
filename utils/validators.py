"""
Validation utilities for input data.
"""

import re
from typing import Tuple, Optional
from urllib.parse import urlparse


def validate_user_intent(user_intent: str) -> Tuple[bool, Optional[str]]:
    """
    Validate user intent for protocol generation.
    
    Args:
        user_intent: User's intent/request
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not user_intent:
        return False, "User intent cannot be empty"
    
    # Check minimum length
    if len(user_intent.strip()) < 10:
        return False, "User intent must be at least 10 characters"
    
    # Check maximum length
    if len(user_intent) > 2000:
        return False, "User intent must be less than 2000 characters"
    
    # Check for potentially harmful content
    harmful_patterns = [
        r'<script',
        r'javascript:',
        r'onerror=',
        r'onclick=',
    ]
    
    user_intent_lower = user_intent.lower()
    for pattern in harmful_patterns:
        if re.search(pattern, user_intent_lower):
            return False, "User intent contains potentially harmful content"
    
    return True, None


def validate_draft_content(draft: str) -> Tuple[bool, Optional[str]]:
    """
    Validate draft protocol content.
    
    Args:
        draft: Protocol draft content
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not draft:
        return False, "Draft content cannot be empty"
    
    # Check minimum length
    if len(draft.strip()) < 100:
        return False, "Draft content must be at least 100 characters"
    
    # Check maximum length (reasonable limit)
    if len(draft) > 50000:
        return False, "Draft content must be less than 50000 characters"
    
    return True, None


def validate_email(email: str) -> bool:
    """
    Validate email address format.
    
    Args:
        email: Email address
        
    Returns:
        True if valid email format
    """
    if not email:
        return False
    
    # Simple email regex
    email_pattern = re.compile(
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    )
    
    return bool(email_pattern.match(email))


def validate_url(url: str) -> bool:
    """
    Validate URL format.
    
    Args:
        url: URL string
        
    Returns:
        True if valid URL
    """
    if not url:
        return False
    
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
    except Exception:
        return False


def validate_thread_id_format(thread_id: str) -> Tuple[bool, Optional[str]]:
    """
    Validate thread ID format with detailed error message.
    
    Args:
        thread_id: Thread identifier
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not thread_id:
        return False, "Thread ID cannot be empty"
    
    # Check UUID4 format
    uuid_pattern = re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$',
        re.IGNORECASE
    )
    
    if not uuid_pattern.match(thread_id):
        return False, "Thread ID must be a valid UUID4"
    
    return True, None


def validate_iteration_count(iteration: int, max_iterations: int) -> Tuple[bool, Optional[str]]:
    """
    Validate iteration count.
    
    Args:
        iteration: Current iteration
        max_iterations: Maximum allowed iterations
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if iteration < 0:
        return False, "Iteration count cannot be negative"
    
    if iteration >= max_iterations:
        return False, f"Iteration count ({iteration}) exceeds maximum ({max_iterations})"
    
    return True, None


def validate_temperature(temperature: float) -> Tuple[bool, Optional[str]]:
    """
    Validate LLM temperature parameter.
    
    Args:
        temperature: Temperature value
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(temperature, (int, float)):
        return False, "Temperature must be a number"
    
    if temperature < 0 or temperature > 2:
        return False, "Temperature must be between 0 and 2"
    
    return True, None


def validate_max_tokens(max_tokens: int) -> Tuple[bool, Optional[str]]:
    """
    Validate max tokens parameter.
    
    Args:
        max_tokens: Maximum tokens value
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(max_tokens, int):
        return False, "Max tokens must be an integer"
    
    if max_tokens < 100:
        return False, "Max tokens must be at least 100"
    
    if max_tokens > 128000:
        return False, "Max tokens cannot exceed 128000"
    
    return True, None
