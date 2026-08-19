#!/usr/bin/env python3
"""
Git Commit Message Generator using Ollama
Automatically generates commit messages from git diff using local LLM models.
"""

import os
import sys
import subprocess
import json
import requests
import argparse
from pathlib import Path
from typing import Optional, Tuple, List
import re

# Try to import jinja2, provide helpful message if not installed
try:
    from jinja2 import Template, Environment, BaseLoader
except ImportError:
    print("Error: jinja2 is required. Install with: pip install jinja2")
    sys.exit(1)


def detect_ollama_model() -> Optional[str]:
    """
    Detect the most recently downloaded/used Ollama model.
    Returns model name or None if no models found.
    """
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        
        lines = result.stdout.strip().split('\n')
        if len(lines) <= 1:  # Only header or empty
            return None
            
        # Parse model list, get the most recent one
        # Format: NAME ID SIZE MODIFIED
        # Skip header
        models = []
        for line in lines[1:]:
            if line.strip():
                parts = line.split()
                if len(parts) >= 4:
                    name = parts[0]
                    # Extract the base model name without tag if present
                    base_name = name.split(':')[0]
                    models.append((base_name, name))
        
        if not models:
            return None
            
        # Return the most recent model (first in list)
        return models[0][1]
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: Ollama not found or not installed.")
        return None


def get_default_prompt_template() -> str:
    """
    Return the default prompt template for generating commit messages.
    """
    return """You are an AI assistant helping to write clear and descriptive git commit messages.

Based on the following git diff output, generate a concise and informative commit message.

Guidelines:
- Use the conventional commit format: <type>(<scope>): <subject>
- Types: feat, fix, docs, style, refactor, perf, test, chore
- Keep the subject line under 50 characters
- Provide a brief description of what changed and why
- If the changes are extensive, include a bulleted list of key changes

Here's the git diff:

{{ diff_output }}

Generate only the commit message, no additional text or explanations."""


def is_git_repository() -> bool:
    """
    Check if current directory is a git repository.
    """
    try:
        subprocess.run(
            ["git", "status"],
            check=True,
            capture_output=True,
            text=True
        )
        return True
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        print("Error: Git not found or not installed.")
        return False


def get_git_diff() -> str:
    """
    Execute git diff and return the output.
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--cached"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error getting git diff: {e}")
        return ""


def generate_commit_message(diff_output: str, model: str, prompt_template: Optional[str] = None) -> Optional[str]:
    """
    Send diff to Ollama and generate commit message.
    """
    if not diff_output.strip():
        print("No staged changes detected. Please stage changes with 'git add'.")
        return None

    # Use template or default
    if prompt_template is None:
        prompt_template = get_default_prompt_template()
    
    # Render template with diff output
    try:
        template = Template(prompt_template)
        prompt = template.render(diff_output=diff_output)
    except Exception as e:
        print(f"Error rendering template: {e}")
        return None

    # Prepare Ollama request
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,  # Lower temperature for more deterministic output
            "top_p": 0.9,
            "max_tokens": 200
        }
    }

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        commit_message = result.get("response", "").strip()
        
        # Clean up the response - remove extra whitespace and quotes
        commit_message = re.sub(r'^["\']|["\']$', '', commit_message)
        
        if not commit_message:
            print("No commit message generated.")
            return None
            
        return commit_message
        
    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to Ollama. Is it running? (http://localhost:11434)")
        return None
    except requests.exceptions.Timeout:
        print("Error: Ollama request timed out.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Ollama: {e}")
        return None


def git_commit(commit_message: str, dry_run: bool = False) -> bool:
    """
    Execute git commit with the generated message.
    """
    if not commit_message:
        return False

    print(f"\nGenerated commit message:")
    print("-" * 50)
    print(commit_message)
    print("-" * 50)

    if dry_run:
        print("Dry run: Not actually committing.")
        return True

    try:
        # Use -m flag to pass the commit message
        result = subprocess.run(
            ["git", "commit", "-m", commit_message],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error committing: {e}")
        print(f"Stderr: {e.stderr}")
        return False


def load_template_from_file(file_path: str) -> Optional[str]:
    """
    Load prompt template from a file.
    """
    try:
        path = Path(file_path)
        if not path.exists():
            print(f"Template file not found: {file_path}")
            return None
        return path.read_text()
    except Exception as e:
        print(f"Error reading template file: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate git commit messages using Ollama LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Use default template and most recent model
  %(prog)s --model llama2     # Use llama2 model specifically
  %(prog)s --template template.j2 # Use custom Jinja2 template file
  %(prog)s --template-inline "{{ diff_output }}" # Use inline template
  %(prog)s --dry-run          # Generate message but don't commit
        """
    )
    
    parser.add_argument(
        "--model",
        help="Ollama model to use (default: auto-detect most recent)"
    )
    parser.add_argument(
        "--template",
        help="Path to Jinja2 template file"
    )
    parser.add_argument(
        "--template-inline",
        help="Inline Jinja2 template string"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate commit message without actually committing"
    )
    
    args = parser.parse_args()

    # Check if git repository
    if not is_git_repository():
        print("Error: Current directory is not a git repository.")
        sys.exit(1)

    # Get model name
    model = args.model
    if not model:
        model = detect_ollama_model()
        if not model:
            print("Error: No Ollama model found. Please specify a model with --model")
            sys.exit(1)
        print(f"Auto-detected model: {model}")

    # Get diff output
    print("Checking for staged changes...")
    diff_output = get_git_diff()
    if not diff_output.strip():
        print("No staged changes found. Stage your changes with 'git add' first.")
        sys.exit(1)

    # Get template
    prompt_template = None
    if args.template and args.template_inline:
        print("Error: Cannot specify both --template and --template-inline")
        sys.exit(1)
    elif args.template:
        prompt_template = load_template_from_file(args.template)
        if prompt_template is None:
            sys.exit(1)
    elif args.template_inline:
        prompt_template = args.template_inline

    # Generate commit message
    print(f"Generating commit message using {model}...")
    commit_message = generate_commit_message(diff_output, model, prompt_template)
    
    if not commit_message:
        print("Failed to generate commit message.")
        sys.exit(1)

    # Perform commit
    if not git_commit(commit_message, args.dry_run):
        sys.exit(1)

    print("✅ Commit completed successfully!")


if __name__ == "__main__":
    main()