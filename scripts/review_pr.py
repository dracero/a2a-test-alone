#!/usr/bin/env python3
"""
Automated PR Reviewer script using Gemini API and .agents/AGENTS.md guidelines.
Can be executed via GitHub Actions or locally in dry-run mode.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import requests

# Importar el detector de errores de cuota centralizado
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from api_key_rotator import _is_quota_error  # noqa: E402

# Try modern google-genai SDK first, fallback to google-generativeai
try:
    from google import genai
    from google.genai import types
    NEW_GENAI_SDK = True
except ImportError:
    try:
        import google.generativeai as legacy_genai
        NEW_GENAI_SDK = False
    except ImportError:
        NEW_GENAI_SDK = None


def load_dotenv_if_exists() -> None:
    """Loads environment variables from local .env file if present."""
    root_dir = Path(__file__).resolve().parent.parent
    env_file = root_dir / ".env"
    if env_file.is_file():
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, val = line.split("=", 1)
                        key = key.strip()
                        val = val.strip().strip("'\"")
                        if key not in os.environ:
                            os.environ[key] = val
        except IOError as err:
            print(f"[Warning] Error reading .env file: {err}", file=sys.stderr)


def get_agents_rules(workspace_root: Path) -> str:
    """Reads project guidelines from .agents/AGENTS.md if available."""
    agents_path = workspace_root / ".agents" / "AGENTS.md"
    if not agents_path.is_file():
        # Fallback to root AGENTS.md if present
        agents_path = workspace_root / "AGENTS.md"

    if agents_path.is_file():
        try:
            with open(agents_path, "r", encoding="utf-8") as f:
                return f.read()
        except IOError as err:
            print(f"[Warning] Could not read AGENTS.md: {err}", file=sys.stderr)

    return "Follow general Python, TypeScript, performance, and clean code best practices."


def fetch_pr_diff(repo: str, pr_number: int, github_token: str) -> str:
    """Fetches the PR diff from GitHub REST API with fallbacks."""
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
    
    # Try standard GitHub diff media types
    for media_type in ["application/vnd.github.diff", "application/vnd.github.patch"]:
        headers = {
            "Accept": media_type,
            "Authorization": f"Bearer {github_token}",
            "User-Agent": "Gemini-PR-Reviewer",
        }
        try:
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 200 and response.text.strip():
                return response.text
        except requests.RequestException as err:
            print(f"[Warning] Failed fetching diff with header {media_type}: {err}", file=sys.stderr)

    # Fallback to web diff URL
    web_diff_url = f"https://github.com/{repo}/pull/{pr_number}.diff"
    try:
        headers = {
            "Authorization": f"Bearer {github_token}",
            "User-Agent": "Gemini-PR-Reviewer",
        }
        response = requests.get(web_diff_url, headers=headers, timeout=30)
        if response.status_code == 200 and response.text.strip():
            return response.text
    except requests.RequestException as err:
        print(f"[Warning] Failed fetching web diff URL: {err}", file=sys.stderr)

    print(f"[Error] Could not retrieve diff for PR #{pr_number} in {repo}.", file=sys.stderr)
    sys.exit(1)


def generate_gemini_review(gemini_keys: Any, rules_text: str, diff_text: str) -> str:
    """Calls Gemini API to perform code review based on AGENTS.md rules with Gemini 3.5 Flash."""
    if NEW_GENAI_SDK is None:
        print("[Error] Neither 'google-genai' nor 'google-generativeai' package is installed.", file=sys.stderr)
        print("Please run: pip install google-genai requests", file=sys.stderr)
        sys.exit(1)

    if isinstance(gemini_keys, str):
        keys_list = [gemini_keys]
    else:
        keys_list = list(gemini_keys)

    system_instruction = (
        "You are an expert AI Code Reviewer. Analyze code changes carefully.\n"
        "Review the provided PR Git diff against the guidelines specified below.\n\n"
        "### GUIDELINES (AGENTS.md):\n"
        f"{rules_text}\n\n"
        "### INSTRUCTIONS FOR YOUR REVIEW:\n"
        "1. Provide a concise Executive Summary.\n"
        "2. Check compliance against each guideline section (General, Python/A2A, GPU/PyTorch, Frontend, Complexity/DB).\n"
        "3. Highlight security vulnerabilities or secret leaks if present.\n"
        "4. Provide actionable, git-diff style recommendations for any issues found.\n"
        "5. Keep feedback clear, direct, and practical."
    )

    prompt_content = f"Please review the following PR code changes (diff):\n\n```diff\n{diff_text}\n```"

    last_err: Optional[Exception] = None
    for idx, key in enumerate(keys_list):
        masked_key = f"...{key[-4:]}" if len(key) >= 4 else "key"
        try:
            if NEW_GENAI_SDK:
                client = genai.Client(api_key=key)
                config = types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2,
                )
                response = client.models.generate_content(
                    model="gemini-3.5-flash",
                    contents=prompt_content,
                    config=config,
                )
                return response.text or "No review response generated."
            else:
                legacy_genai.configure(api_key=key)
                model = legacy_genai.GenerativeModel(
                    model_name="gemini-3.5-flash",
                    system_instruction=system_instruction,
                )
                response = model.generate_content(
                    prompt_content,
                    generation_config={"temperature": 0.2},
                )
                return response.text or "No review response generated."

        except Exception as err:
            last_err = err
            err_str = str(err).lower()
            if _is_quota_error(err) and idx < len(keys_list) - 1:
                print(f"[Warning] Key {masked_key} falló ({type(err).__name__}). Reintentando con siguiente key disponible...", file=sys.stderr)
                continue
            else:
                break

    print(f"[Error] Gemini API invocation failed: {last_err}", file=sys.stderr)
    sys.exit(1)


def post_pr_comment(repo: str, pr_number: int, github_token: str, review_body: str) -> None:
    """Posts review as a comment on the GitHub PR."""
    url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"Bearer {github_token}",
        "User-Agent": "Gemini-PR-Reviewer",
    }
    header_text = "## 🤖 Automated Code Review (Gemini 3.5 Flash & AGENTS.md)\n\n"
    payload: Dict[str, Any] = {"body": header_text + review_body}

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        print(f"[Success] Successfully posted review comment on PR #{pr_number}.")
    except requests.RequestException as err:
        print(f"[Error] Failed to post comment on GitHub PR: {err}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    load_dotenv_if_exists()

    parser = argparse.ArgumentParser(description="Automated PR Reviewer using Gemini 3.5 Flash & AGENTS.md")
    parser.add_argument("--repo", type=str, default=os.getenv("GITHUB_REPOSITORY"), help="GitHub repository (owner/repo)")
    parser.add_argument("--pr", type=int, default=int(os.getenv("PR_NUMBER", "0")) if os.getenv("PR_NUMBER") else 0, help="PR number")
    parser.add_argument("--diff-file", type=str, help="Path to local diff file for dry-run testing")
    parser.add_argument("--dry-run", action="store_true", help="Print review output to stdout without posting to GitHub")
    args = parser.parse_args()

    # Collect available candidate keys (GEMINI_API_KEY, GOOGLE_API_KEY or GOOGLE_API_KEYS)
    candidate_keys = []
    if os.getenv("GEMINI_API_KEY"):
        candidate_keys.append(os.getenv("GEMINI_API_KEY").strip())
    if os.getenv("GOOGLE_API_KEY"):
        candidate_keys.append(os.getenv("GOOGLE_API_KEY").strip())
    if os.getenv("GOOGLE_API_KEYS"):
        for k in os.getenv("GOOGLE_API_KEYS", "").split(","):
            k_clean = k.strip()
            if k_clean and k_clean not in candidate_keys:
                candidate_keys.append(k_clean)

    if not candidate_keys:
        print("[Error] No Gemini API key found. Set GEMINI_API_KEY, GOOGLE_API_KEY or GOOGLE_API_KEYS.", file=sys.stderr)
        sys.exit(1)

    workspace_root = Path(__file__).resolve().parent.parent
    rules_text = get_agents_rules(workspace_root)

    diff_text = ""
    if args.diff_file:
        diff_path = Path(args.diff_file)
        if not diff_path.is_file():
            print(f"[Error] Diff file '{args.diff_file}' not found.", file=sys.stderr)
            sys.exit(1)
        diff_text = diff_path.read_text(encoding="utf-8")
    elif args.repo and args.pr > 0:
        github_token = os.getenv("GITHUB_TOKEN")
        if not github_token and not args.dry_run:
            print("[Error] GITHUB_TOKEN environment variable is required to fetch PR diff and post comments.", file=sys.stderr)
            sys.exit(1)
        token = github_token or "dummy_token"
        diff_text = fetch_pr_diff(args.repo, args.pr, token)
    else:
        print("[Error] Must specify --repo and --pr (or set GITHUB_REPOSITORY and PR_NUMBER), or pass --diff-file.", file=sys.stderr)
        sys.exit(1)

    if not diff_text.strip():
        print("[Info] Diff is empty. No code changes to review.")
        sys.exit(0)

    print(f"[Info] Analyzing code changes with Gemini 3.5 Flash ({len(candidate_keys)} key(s) available)...")
    review_output = generate_gemini_review(candidate_keys, rules_text, diff_text)

    if args.dry_run:
        print("\n--- [DRY RUN REVIEW OUTPUT] ---")
        print(review_output)
        print("--- [END DRY RUN] ---\n")
    else:
        github_token = os.getenv("GITHUB_TOKEN")
        if not github_token:
            print("[Error] GITHUB_TOKEN environment variable is required to post comments.", file=sys.stderr)
            sys.exit(1)
        post_pr_comment(args.repo, args.pr, github_token, review_output)


if __name__ == "__main__":
    main()
