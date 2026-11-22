"""
Core logic for the Library Assistant Bot (PS15).

Pipeline:  user query -> keyword filter on the catalogue (pandas)
           -> build prompt with the matching rows -> LLM picks & explains top 3
"""

import os
import re

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Columns that are searched by the keyword filter
SEARCH_COLUMNS = ["title", "author", "topic", "description"]

# Max number of matching rows sent to the LLM (keeps the prompt short)
MAX_ROWS_TO_LLM = 15

# Common words that should not count as search keywords
STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "on", "in", "to", "for", "by", "with",
    "about", "is", "are", "am", "i", "me", "my", "we", "you", "your", "it",
    "book", "books", "recommend", "recommendation", "suggest", "suggestions",
    "want", "need", "looking", "find", "show", "give", "some", "any", "good",
    "best", "please", "can", "could", "would", "like", "read", "reading",
    "learn", "study", "topic", "topics", "author", "written",
    "writer", "something", "related", "beginner", "beginners", "advanced",
    "easy", "simple", "what", "which", "has", "have", "there", "that", "this",
}

SYSTEM_PROMPT = """You are a friendly college library assistant.
You help students choose books ONLY from the catalogue rows given to you.

Rules:
1. Recommend up to 3 books, ranked best match first. Only include books that clearly match the request — it is fine to recommend fewer than 3.
2. Use ONLY books from the provided catalogue rows. Never invent books, authors or shelf numbers.
3. For each book give: title, author, shelf, availability, and 1-2 lines on why it fits the student's request.
4. If a book is not available (available = No), still mention it if it is a strong match, but say it is currently issued and suggest an available alternative if one exists.
5. If none of the rows really fit the request, say so honestly instead of forcing a recommendation.
6. Keep the answer short and use this format:

**1. <Title>** by <Author>
Shelf: <shelf> | Available: <Yes/No>
Why: <reason>

(repeat for 2 and 3, then one short closing line)"""


def load_catalogue(path_or_file) -> pd.DataFrame:
    """Load the catalogue CSV and make sure the required columns exist."""
    df = pd.read_csv(path_or_file)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = {"title", "author"} - set(df.columns)
    if missing:
        raise ValueError(f"Catalogue is missing required column(s): {', '.join(missing)}")
    return df.fillna("")


def extract_keywords(query: str) -> list[str]:
    """Turn the user's message into a list of search keywords."""
    words = re.findall(r"[a-zA-Z0-9+#]+", query.lower())
    return [w for w in words if w not in STOPWORDS and len(w) >= 2]


def filter_catalogue(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """
    Keyword filter: score each book by how many keywords appear in its
    title / author / topic / description, and return the best matches.
    """
    keywords = extract_keywords(query)
    if not keywords:
        return df.iloc[0:0]

    columns = [c for c in SEARCH_COLUMNS if c in df.columns]
    text = df[columns].astype(str).agg(" ".join, axis=1).str.lower()

    score = pd.Series(0, index=df.index)
    for kw in keywords:
        # Whole-word match at the start (so "network" matches "networks",
        # but "java" does not match inside another word)
        pattern = r"\b" + re.escape(kw)
        score += text.str.contains(pattern, regex=True).astype(int)

    # Bonus if the whole phrase appears together, e.g. "machine learning"
    phrase = " ".join(keywords)
    if len(keywords) > 1:
        score += text.str.contains(re.escape(phrase), regex=True).astype(int) * 2

    result = df.assign(match_score=score)
    result = result[result["match_score"] > 0]
    return result.sort_values("match_score", ascending=False).head(MAX_ROWS_TO_LLM)


def build_prompt(query: str, matches: pd.DataFrame) -> str:
    """Put the user's request and the matching catalogue rows into one prompt."""
    rows_csv = matches.drop(columns=["match_score"], errors="ignore").to_csv(index=False)
    return (
        f"Student's request: {query}\n\n"
        f"Matching catalogue rows (CSV):\n{rows_csv}\n"
        "Recommend and explain the top 3 books for this student."
    )


def get_client() -> tuple[OpenAI, str]:
    """
    Create an LLM client. Groq, Gemini and Ollama all offer an
    OpenAI-compatible API, so one client works for all of them.
    Settings come from the .env file.
    """
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "groq":
        base_url = "https://api.groq.com/openai/v1"
        api_key = os.getenv("GROQ_API_KEY")
        model = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
    elif provider == "gemini":
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        api_key = os.getenv("GEMINI_API_KEY")
        model = os.getenv("LLM_MODEL", "gemini-2.5-flash")
    elif provider == "ollama":
        base_url = "http://localhost:11434/v1"
        api_key = "ollama"  # Ollama ignores the key but the client needs one
        model = os.getenv("LLM_MODEL", "llama3.2")
    else:
        raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use groq, gemini or ollama.")

    if not api_key:
        raise ValueError(f"API key for '{provider}' not found. Add it to your .env file.")

    return OpenAI(base_url=base_url, api_key=api_key), model


def recommend(query: str, matches: pd.DataFrame) -> str:
    """Send the prompt to the LLM and return its answer."""
    client, model = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(query, matches)},
        ],
        temperature=0.3,  # low temperature = more consistent, factual answers
    )
    return response.choices[0].message.content
