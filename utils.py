def clean_json(response: str) -> str:
    """Remove any markdown json codeblock and newline formatting if present. Useful for non schema enforceable models"""
    cleaned = response.strip().replace("\n", "").removeprefix("```json").removesuffix("```")
    
    # Remove whitespace while preserving JSON structure
    import re
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    
    # Remove trailing commas in arrays, preserving the already cleaned string
    cleaned = re.sub(r',(\s*)]', r'\1]', cleaned)
    cleaned = cleaned.replace(",]", "]")
    return cleaned

