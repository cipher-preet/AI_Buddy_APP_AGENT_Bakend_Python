from pathlib import Path


PROMPT_DIR = Path(__file__).parent


def load_prompt(prompt_name: str) -> str:
    path = PROMPT_DIR / f"{prompt_name}.md"
    return path.read_text(encoding="utf-8")
