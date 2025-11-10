"""RAG suggestion system using OpenAI GPT-5/GPT-4o."""
import os
from typing import List, Dict
from ..config import settings

# All possible checks
ALL_CHECKS = ["Crosswalk Error", "Banned Phrases", "Name Inconsistency"]


def generate_rag_suggestions(text: str, failed_checks: List[str]) -> str:
    """
    Generate AI-powered fix suggestions using GPT-5/GPT-4o (or mock response).

    Args:
        text: Full proposal text
        failed_checks: List of failed check names

    Returns:
        HTML-formatted suggestions
    """
    # If no failures, return success message
    if not failed_checks:
        return "<p>✓ All compliance checks passed. No fixes needed.</p>"

    # Read API key
    api_key = settings.openai_api_key

    # Return mock suggestions if using dummy key
    if api_key.startswith("sk-dummy"):
        return _generate_mock_suggestions_html(failed_checks)

    # Real GPT implementation
    try:
        return _call_openai_gpt(api_key, text, failed_checks)
    except Exception as e:
        return f"<p>⚠ Error generating suggestions: {str(e)}. Please review manually.</p>"


def _generate_mock_suggestions_html(failed_checks: List[str]) -> str:
    """Generate mock HTML suggestions for testing."""
    html_parts = []
    check_number = 1
    
    # Base styles for neo-brutalism theme
    base_style = "font-family: 'DM Sans', sans-serif; color: #000000; margin: 0; padding: 0;"
    div_style = f"{base_style} margin-bottom: 1rem; padding: 0.75rem; border: 2px solid #000000; background-color: #ffffff;"
    strong_style = f"{base_style} font-weight: 700; font-size: 1rem; text-transform: uppercase;"
    ul_style = f"{base_style} margin: 0.5rem 0; padding-left: 1.5rem; list-style: none;"
    li_style = f"{base_style} margin: 0.5rem 0; padding-left: 0;"
    hr_style = "margin: 1.5rem 0; border: none; border-top: 2px solid #000000; height: 2px; background-color: #000000;"
    p_style = f"{base_style} margin: 0.5rem 0; font-size: 0.9rem;"
    
    # Generate failed checks
    for check in failed_checks:
        if check == "Crosswalk Error":
            html_parts.append(f'<div style="{div_style}"><strong style="{strong_style}">Failed Check {check_number}: Crosswalk Error:</strong>')
            html_parts.append(f'<ul style="{ul_style}">')
            html_parts.append(f'<li style="{li_style}"><strong style="font-weight: 600;">Section:</strong> EXECUTIVE SUMMARY</li>')
            html_parts.append(f'<li style="{li_style}"><strong style="font-weight: 600;">Quote:</strong> "Requirement R1 is detailed in Section 5: Work Approach."</li>')
            html_parts.append(f'<li style="{li_style}"><strong style="font-weight: 600;">Problem:</strong> Requirement citations may be incorrect</li>')
            html_parts.append(f'<li style="{li_style}"><strong style="font-weight: 600;">Recommendation:</strong> Verify that R1-R5 citations match the correct RFP sections</li>')
            html_parts.append('</ul></div>')
        elif check == "Banned Phrases":
            html_parts.append(f'<div style="{div_style}"><strong style="{strong_style}">Failed Check {check_number}: Banned Phrases:</strong>')
            html_parts.append(f'<ul style="{ul_style}">')
            html_parts.append(f'<li style="{li_style}"><strong style="font-weight: 600;">Section:</strong> TECHNICAL APPROACH</li>')
            html_parts.append(f'<li style="{li_style}"><strong style="font-weight: 600;">Quote:</strong> "We guarantee unconditional savings..."</li>')
            html_parts.append(f'<li style="{li_style}"><strong style="font-weight: 600;">Problem:</strong> Prohibited language detected</li>')
            html_parts.append(f'<li style="{li_style}"><strong style="font-weight: 600;">Recommendation:</strong> Replace with compliant phrasing (e.g., "We will make every effort to...")</li>')
            html_parts.append('</ul></div>')
        elif check == "Name Inconsistency":
            html_parts.append(f'<div style="{div_style}"><strong style="{strong_style}">Failed Check {check_number}: Name Inconsistency:</strong>')
            html_parts.append(f'<ul style="{ul_style}">')
            html_parts.append(f'<li style="{li_style}"><strong style="font-weight: 600;">Section:</strong> EXECUTIVE SUMMARY and TEAM QUALIFICATIONS</li>')
            html_parts.append(f'<li style="{li_style}"><strong style="font-weight: 600;">Quote:</strong> "John Smith" vs "John S."</li>')
            html_parts.append(f'<li style="{li_style}"><strong style="font-weight: 600;">Problem:</strong> PM name appears in different formats</li>')
            html_parts.append(f'<li style="{li_style}"><strong style="font-weight: 600;">Recommendation:</strong> Use consistent full name throughout the proposal</li>')
            html_parts.append('</ul></div>')
        
        check_number += 1
        if check_number <= len(failed_checks):
            html_parts.append(f'<hr style="{hr_style}">')
    
    # Add passed checks
    passed_checks = [check for check in ALL_CHECKS if check not in failed_checks]
    if passed_checks:
        html_parts.append(f'<div style="{base_style} margin-top: 1rem;">')
        for check in passed_checks:
            html_parts.append(f'<p style="{p_style}">{check}: No problem here.</p>')
        html_parts.append('</div>')
    
    html_parts.append(f'<p style="{base_style} margin-top: 1rem; font-style: italic; font-size: 0.85rem; opacity: 0.7;">Note: Using mock suggestions. Configure GPT API key for detailed analysis.</p>')
    return ''.join(html_parts)


def _call_openai_gpt(api_key: str, text: str, failed_checks: List[str]) -> str:
    """Call OpenAI GPT-5/GPT-4o for real suggestions."""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        # Determine passed checks
        passed_checks = [check for check in ALL_CHECKS if check not in failed_checks]

        system_prompt = """You are an HDR proposal compliance expert. Analyze proposals for these issues:

1. Crosswalk Errors: Wrong requirement IDs (R1-R5) cited in wrong sections
2. Banned Phrases: "unconditional guarantee", "guaranteed savings", "risk-free delivery", "absolute assurance"
3. Name Inconsistency: PM name in different formats (e.g., "John Smith" vs "John S.")

For each FAILED check, provide HTML output in this exact format with neo-brutalism styling:
<div style="font-family: 'DM Sans', sans-serif; color: #000000; margin: 0; padding: 0; margin-bottom: 1rem; padding: 0.75rem; border: 2px solid #000000; background-color: #ffffff;">
<strong style="font-family: 'DM Sans', sans-serif; color: #000000; margin: 0; padding: 0; font-weight: 700; font-size: 1rem; text-transform: uppercase;">Failed Check [NUMBER]: [CHECK NAME]:</strong>
<ul style="font-family: 'DM Sans', sans-serif; color: #000000; margin: 0; padding: 0; margin: 0.5rem 0; padding-left: 1.5rem; list-style: none;">
<li style="font-family: 'DM Sans', sans-serif; color: #000000; margin: 0; padding: 0; margin: 0.5rem 0; padding-left: 0;"><strong style="font-weight: 600;">Section:</strong> [Section name where issue appears]</li>
<li style="font-family: 'DM Sans', sans-serif; color: #000000; margin: 0; padding: 0; margin: 0.5rem 0; padding-left: 0;"><strong style="font-weight: 600;">Quote:</strong> "[Exact quote from the proposal]"</li>
<li style="font-family: 'DM Sans', sans-serif; color: #000000; margin: 0; padding: 0; margin: 0.5rem 0; padding-left: 0;"><strong style="font-weight: 600;">Problem:</strong> [Clear explanation of the problem]</li>
<li style="font-family: 'DM Sans', sans-serif; color: #000000; margin: 0; padding: 0; margin: 0.5rem 0; padding-left: 0;"><strong style="font-weight: 600;">Recommendation:</strong> [Actionable fix recommendation]</li>
</ul>
</div>
<hr style="margin: 1.5rem 0; border: none; border-top: 2px solid #000000; height: 2px; background-color: #000000;">

For PASSED checks, simply output:
<p style="font-family: 'DM Sans', sans-serif; color: #000000; margin: 0; padding: 0; margin: 0.5rem 0; font-size: 0.9rem;">[CHECK NAME]: No problem here.</p>

Use "Section" not "Location". Only emphasize failed checks with <strong> tags. Separate failed checks with <hr> tags. Use neo-brutalism styling: bold black borders (2px solid #000000), no rounded corners, stark typography, black text on white background."""

        user_prompt = f"""This proposal failed these compliance checks: {', '.join(failed_checks)}
Passed checks: {', '.join(passed_checks) if passed_checks else 'None'}

Analyze the full proposal text below and provide HTML-formatted fix suggestions:

{text}"""

        # Try GPT-5 first, fallback to gpt-4o if not available
        models_to_try = ["gpt-5", "gpt-4o"]
        
        for model_name in models_to_try:
            try:
                params = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                }
                
                # GPT-5 uses max_completion_tokens, GPT-4o uses max_tokens
                if model_name == "gpt-5":
                    params["max_completion_tokens"] = 1000
                else:
                    params["max_tokens"] = 1000
                    params["temperature"] = 0.3
                
                print(f"🔍 Using {model_name} model")
                response = client.chat.completions.create(**params)
                print(f"✓ Successfully used {model_name}")

                if response.choices and len(response.choices) > 0:
                    message = response.choices[0].message
                    content = getattr(message, 'content', None) if message else None

                    if content and content.strip():
                        # Ensure we have proper HTML structure
                        html_content = content.strip()
                        # If GPT didn't format passed checks, add them
                        if passed_checks:
                            passed_html = '<div style="margin-top: 1rem;">'
                            for check in passed_checks:
                                if check not in html_content:
                                    passed_html += f'<p>{check}: No problem here.</p>'
                            passed_html += '</div>'
                            if passed_html != '<div style="margin-top: 1rem;"></div>':
                                html_content += passed_html
                        return html_content
                
            except Exception as e:
                error_msg = str(e).lower()
                if "model" in error_msg or "not found" in error_msg or "invalid" in error_msg:
                    print(f"⚠ Model {model_name} not available, trying next...")
                    continue
                raise
        
        # If all models failed, return mock
        return _generate_mock_suggestions_html(failed_checks)

    except ImportError:
        return "<p>⚠ OpenAI library not installed. Using mock suggestions instead.</p>" + _generate_mock_suggestions_html(failed_checks)
    except Exception as e:
        raise Exception(f"OpenAI API call failed: {str(e)}")
