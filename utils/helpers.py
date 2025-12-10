"""
Helper functions for common operations.
"""

import re
import json
import hashlib
from uuid import uuid4
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta


def generate_thread_id() -> str:
    """
    Generate a unique thread ID.
    
    Returns:
        UUID4 string
    """
    return str(uuid4())


def sanitize_text(text: str, max_length: Optional[int] = None) -> str:
    """
    Sanitize text by removing potentially harmful content.
    
    Args:
        text: Text to sanitize
        max_length: Optional maximum length
        
    Returns:
        Sanitized text
    """
    if not text:
        return ""
    
    # Remove control characters except newline and tab
    sanitized = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
    
    # Trim whitespace
    sanitized = sanitized.strip()
    
    # Apply max length if specified
    if max_length and len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length with suffix.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncated
        
    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def calculate_word_count(text: str) -> int:
    """
    Calculate word count in text.
    
    Args:
        text: Text to count
        
    Returns:
        Word count
    """
    if not text:
        return 0
    
    # Split on whitespace and filter empty strings
    words = [word for word in text.split() if word.strip()]
    return len(words)


def extract_sections(text: str) -> Dict[str, str]:
    """
    Extract markdown sections from text.
    
    Args:
        text: Text with markdown headers
        
    Returns:
        Dictionary mapping section names to content
    """
    sections = {}
    current_section = "introduction"
    current_content = []
    
    lines = text.split('\n')
    
    for line in lines:
        # Check if line is a header (starts with # or ##)
        if line.strip().startswith('#'):
            # Save previous section
            if current_content:
                sections[current_section] = '\n'.join(current_content).strip()
            
            # Start new section
            current_section = line.strip('#').strip().lower().replace(' ', '_')
            current_content = []
        else:
            current_content.append(line)
    
    # Save last section
    if current_content:
        sections[current_section] = '\n'.join(current_content).strip()
    
    return sections


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 1:
        return f"{int(seconds * 1000)}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def validate_thread_id(thread_id: str) -> bool:
    """
    Validate thread ID format (UUID4).
    
    Args:
        thread_id: Thread ID to validate
        
    Returns:
        True if valid UUID4
    """
    uuid_pattern = re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$',
        re.IGNORECASE
    )
    return bool(uuid_pattern.match(thread_id))


def parse_boolean(value: Any) -> bool:
    """
    Parse various boolean representations.
    
    Args:
        value: Value to parse
        
    Returns:
        Boolean value
    """
    if isinstance(value, bool):
        return value
    
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes', 'on', 'enabled')
    
    if isinstance(value, (int, float)):
        return value != 0
    
    return False


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """
    Safely parse JSON string with fallback.
    
    Args:
        json_str: JSON string to parse
        default: Default value if parsing fails
        
    Returns:
        Parsed JSON or default value
    """
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError, ValueError):
        return default


def hash_text(text: str, algorithm: str = "sha256") -> str:
    """
    Generate hash of text.
    
    Args:
        text: Text to hash
        algorithm: Hash algorithm (md5, sha1, sha256)
        
    Returns:
        Hex digest of hash
    """
    hasher = hashlib.new(algorithm)
    hasher.update(text.encode('utf-8'))
    return hasher.hexdigest()


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to chunk
        chunk_size: Size of each chunk
        overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    
    return chunks


def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate simple similarity between two texts.
    
    Uses Jaccard similarity of word sets.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score (0.0 to 1.0)
    """
    if not text1 or not text2:
        return 0.0
    
    # Convert to word sets
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # Calculate Jaccard similarity
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    if not union:
        return 0.0
    
    return len(intersection) / len(union)


def extract_urls(text: str) -> List[str]:
    """
    Extract URLs from text.
    
    Args:
        text: Text to extract from
        
    Returns:
        List of URLs found
    """
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    return url_pattern.findall(text)


def format_timestamp(dt: datetime, format_type: str = "iso") -> str:
    """
    Format datetime in various formats.
    
    Args:
        dt: Datetime object
        format_type: Format type (iso, human, short)
        
    Returns:
        Formatted timestamp
    """
    if format_type == "iso":
        return dt.isoformat()
    elif format_type == "human":
        return dt.strftime("%B %d, %Y at %I:%M %p")
    elif format_type == "short":
        return dt.strftime("%Y-%m-%d %H:%M")
    else:
        return str(dt)


def time_ago(dt: datetime) -> str:
    """
    Convert datetime to human-readable "time ago" format.
    
    Args:
        dt: Datetime object
        
    Returns:
        Human-readable time ago string
    """
    now = datetime.now()
    diff = now - dt
    
    seconds = diff.total_seconds()
    
    if seconds < 60:
        return "just now"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    elif seconds < 86400:
        hours = int(seconds // 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif seconds < 604800:
        days = int(seconds // 86400)
        return f"{days} day{'s' if days != 1 else ''} ago"
    elif seconds < 2592000:
        weeks = int(seconds // 604800)
        return f"{weeks} week{'s' if weeks != 1 else ''} ago"
    elif seconds < 31536000:
        months = int(seconds // 2592000)
        return f"{months} month{'s' if months != 1 else ''} ago"
    else:
        years = int(seconds // 31536000)
        return f"{years} year{'s' if years != 1 else ''} ago"


def batch_items(items: List[Any], batch_size: int) -> List[List[Any]]:
    """
    Batch items into smaller groups.
    
    Args:
        items: List of items to batch
        batch_size: Size of each batch
        
    Returns:
        List of batches
    """
    batches = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i:i + batch_size])
    return batches


def deep_merge(dict1: Dict, dict2: Dict) -> Dict:
    """
    Deep merge two dictionaries.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary (takes precedence)
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result
