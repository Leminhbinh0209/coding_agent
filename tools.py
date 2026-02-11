from typing import TypedDict
import sys
from io import StringIO
import os
from typing import Optional, Dict, Any
class ToolError(Exception):
    """Custom exception for tool failures"""
    pass


class Execution(TypedDict):
    results: list[str]
    errors: list[str]

def execute_code(code: str) -> Execution:
    execution = {"results": [], "errors": []}
    old_stdout = sys.stdout
    try:
        sys.stdout = StringIO()
        exec(code)
        result = sys.stdout.getvalue()
        sys.stdout = old_stdout
        execution["results"] = [result]
    except Exception as e:
        execution["errors"] = [str(e)]
    finally:
        sys.stdout = old_stdout
        return execution

execute_code_schema = {
    "type": "function",
    "function": { 
        "name": "execute_code",
        "description": "Execute Python code and return the result or error.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute as a string",
                }
            },
            "required": ["code"],
            "additionalProperties": False,
        },
    }
}
def read_file(
    file_path: str, limit: Optional[int] = None, offset: int = 0
) -> Dict[str, Any]:
    """Read file content with optional offset and limit."""
    if not os.path.exists(file_path):
        raise ToolError(f"File does not exist: {file_path}")

    if not os.path.isfile(file_path):
        raise ToolError(f"Path is not a file: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            if offset > 0:
                f.seek(offset)
            content = f.read(limit) if limit else f.read()

        return {"content": content, "size": len(content)}

    except PermissionError:
        raise ToolError(f"Permission denied: {file_path}")
    except UnicodeDecodeError:
        raise ToolError(f"Cannot decode file as UTF-8: {file_path}")
read_file_schema = {
    "type": "function",
    "function": { 
        "name": "read_file",
        "description": "Read file content with optional offset and limit.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the file to read",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of characters to read",
                },
                "offset": {
                    "type": "integer",
                    "description": "Starting position in the file",
                },
            },
            "required": ["file_path"],
            "additionalProperties": False,
        },
    }
}
def write_file(content: str, file_path: str) -> Dict[str, Any]:
    """Write content to file, creating directories if needed."""
    try:
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        try:
            # Try to decode if it looks like it has escape sequences
            if '\\n' in content or '\\t' in content:
                content = content.encode().decode('unicode_escape')
        except:
            # If decode fails, use content as-is
            pass
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        file_size = os.path.getsize(file_path)
        # we keep the result minimal
        return {
            "message": f"Written {file_size} bytes to {file_path}",
            "size": file_size,
        }

    except PermissionError:
        raise ToolError(f"Permission denied: {file_path}")
write_file_schema = {
    "type": "function",
    "function": { 
        "name": "write_file",
        "description": "Write content to file, creating directories if needed.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Content to write to the file",
                },
                "file_path": {
                    "type": "string",
                    "description":  "Absolute path where the file will be written",
                },
            },
            "required": ["content", "file_path"],
            "additionalProperties": False,
        },
    }
}