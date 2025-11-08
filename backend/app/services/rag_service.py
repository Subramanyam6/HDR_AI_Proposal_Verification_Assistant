"""RAG suggestion system using OpenAI GPT-4o."""
import os
from typing import List, Dict
from ..config import settings


def generate_rag_suggestions(text: str, failed_checks: List[str]) -> str:
    """
    Generate AI-powered fix suggestions using GPT-4o (or mock response).

    Args:
        text: Full proposal text
        failed_checks: List of failed check names

    Returns:
        Markdown-formatted suggestions
    """
    # If no failures, return success message
    if not failed_checks:
        return "✓ All compliance checks passed. No fixes needed."

    # Read API key
    api_key = settings.openai_api_key

    # Return mock suggestions if using dummy key
    if api_key.startswith("sk-dummy"):
        return _generate_mock_suggestions(failed_checks)

    # Real GPT-4o implementation
    try:
        return _call_openai_gpt4o(api_key, text, failed_checks)
    except Exception as e:
        return f"⚠ Error generating suggestions: {str(e)}. Please review manually."


def _generate_mock_suggestions(failed_checks: List[str]) -> str:
    """Generate mock suggestions for testing."""
    mock_output = ""

    for check in failed_checks:
        if check == "Crosswalk Error":
            mock_output += "**✗ Crosswalk Error**\n\n"
            mock_output += "- **Issue Found:** Requirement citations may be incorrect\n"
            mock_output += "- **Recommended Fix:** Verify that R1-R5 citations match the correct RFP sections\n\n"
        elif check == "Banned Phrases":
            mock_output += "**✗ Banned Phrases**\n\n"
            mock_output += "- **Issue Found:** Prohibited language detected\n"
            mock_output += "- **Recommended Fix:** Replace with compliant phrasing (e.g., 'We will make every effort to...')\n\n"
        elif check == "Name Inconsistency":
            mock_output += "**✗ Name Inconsistency**\n\n"
            mock_output += "- **Issue Found:** PM name appears in different formats\n"
            mock_output += "- **Recommended Fix:** Use consistent full name throughout the proposal\n\n"

    mock_output += "*Note: Using mock suggestions. Configure GPT-4o API key for detailed analysis.*"
    return mock_output.strip()


def _call_openai_gpt4o(api_key: str, text: str, failed_checks: List[str]) -> str:
    """Call OpenAI GPT-4o for real suggestions."""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        system_prompt = """You are an HDR proposal compliance expert. Analyze proposals for these issues:

1. Crosswalk Errors: Wrong requirement IDs (R1-R5) cited in wrong sections
2. Banned Phrases: "unconditional guarantee", "guaranteed savings", "risk-free delivery", "absolute assurance"
3. Name Inconsistency: PM name in different formats (e.g., "John Smith" vs "John S.")

For each failed check, provide:
- Specific location where issue appears (quote the text)
- Clear explanation of the problem
- Actionable fix recommendation

Format output with Markdown headers and bullet points."""

        user_prompt = f"""This proposal failed these compliance checks: {', '.join(failed_checks)}

Analyze the full proposal text below and provide specific fix suggestions:

{text}"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=1000,
        )

        content = response.choices[0].message.content
        return content.strip() if content else "No suggestions returned by the language model."

    except ImportError:
        return "⚠ OpenAI library not installed. Using mock suggestions instead.\n\n" + _generate_mock_suggestions(failed_checks)
    except Exception as e:
        raise Exception(f"OpenAI API call failed: {str(e)}")
