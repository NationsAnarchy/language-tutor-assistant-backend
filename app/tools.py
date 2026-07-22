"""
LangChain function-calling tools for the Language Tutor Agent.

Tools exposed to the LLM:
- retrieve_grammar: Query Pinecone for grammar notes
- retrieve_vocab: Query Pinecone for vocabulary entries
- generate_exercise: Build a quiz item from retrieved context
- grade_answer: Grade a submitted exercise answer (Week 2)
- log_mistake: Log a corrected mistake to the session (Week 2)

Error handling:
  Each tool wraps its LLM / retrieval call in try/except. On failure, the tool
  returns a structured fallback string the agent can read and continue with,
  rather than raising an exception that would break the graph.
"""

import json
import os
from contextvars import ContextVar

from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from .logging_config import get_logger

logger = get_logger(__name__)


def _extract_text(content: object) -> str:
    """Normalize Gemini's list-of-content-parts into a plain string.

    Gemini returns content as [{'type': 'text', 'text': '...'}] but the
    rest of the codebase expects plain strings. This helper handles both
    formats transparently.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(part.get("text", ""))
            elif hasattr(part, "text"):
                parts.append(part.text)
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content)

# contextvars for session context — tools can access these without explicit parameter passing
# ponytail: shared mutable context per request, safe because graph.invoke() is sync and
# FastAPI runs sync endpoints in a threadpool (one thread per request).
_CURRENT_USER_ID: ContextVar[str] = ContextVar("current_user_id", default="")
_CURRENT_SESSION_ID: ContextVar[str] = ContextVar("current_session_id", default="")


# ---------------------------------------------------------------------------
# Stateful retriever holder — initialized at app startup
# ---------------------------------------------------------------------------
_vector_store: PineconeVectorStore | None = None


def init_vector_store(pinecone_index, api_key: str) -> None:
    """Initialize the LangChain PineconeVectorStore with the given index.

    Uses GOOGLE_EMBEDDING_API_KEY if set, falling back to the provided api_key (GEMINI_API_KEY).
    """
    global _vector_store
    embedding_api_key = os.getenv("GOOGLE_EMBEDDING_API_KEY") or api_key
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=embedding_api_key,
    )
    _vector_store = PineconeVectorStore(
        index=pinecone_index,
        embedding=embeddings,
    )


class _EmptyRetriever:
    """Stub retriever that returns no results — used when the vector store is unavailable."""

    def invoke(self, query: str, **kwargs):
        return []


def _get_retriever(language: str, level: str | None = None, topic: str | None = None, k: int = 3):
    """Build a retriever with metadata filters.

    Returns a stub retriever that yields empty results if the vector store is not initialized,
    so tools can degrade gracefully instead of raising.
    """
    if _vector_store is None:
        logger.warning("Vector store not initialized — returning empty retriever")
        return _EmptyRetriever()

    filter_dict: dict = {"language": language}
    if level:
        filter_dict["level"] = level
    if topic:
        filter_dict["topic"] = topic

    return _vector_store.as_retriever(
        search_kwargs={"k": k, "filter": filter_dict, "namespace": language},
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def retrieve_grammar(language: str, topic: str, level: str = "beginner") -> str:
    """Retrieve grammar rules and explanations for the given language, topic, and level.

    Args:
        language: Target language code — "en", "ko", or "ja"
        topic: Grammar topic to search for (e.g., "tenses", "particles", "conditionals")
        level: Learner level — "beginner", "intermediate", or "advanced"

    Returns:
        Formatted string of retrieved grammar notes, or a message indicating no results.
    """
    try:
        retriever = _get_retriever(language, level=level, topic=topic, k=3)
        docs = retriever.invoke(f"{topic} grammar {level}")
    except Exception as exc:
        logger.warning("retrieve_grammar failed for %s/%s: %s", language, topic, exc)
        return f"(retrieval temporarily unavailable for grammar topic '{topic}')"

    if not docs:
        return "(no retrieved grammar notes available for this topic)"

    results = []
    for doc in docs:
        title = doc.metadata.get("title", doc.metadata.get("topic", "Grammar Note"))
        results.append(f"**{title}**\n{doc.page_content}")

    return "\n\n---\n\n".join(results)


@tool
def retrieve_vocab(language: str, topic_or_word: str, level: str = "beginner") -> str:
    """Retrieve vocabulary lists and usage examples for the given language and topic.

    Args:
        language: Target language code — "en", "ko", or "ja"
        topic_or_word: Topic or specific word to search for (e.g., "food", "greetings", "emotions")
        level: Learner level — "beginner", "intermediate", or "advanced"

    Returns:
        Formatted string of retrieved vocabulary, or a message indicating no results.
    """
    try:
        retriever = _get_retriever(language, level=level, topic=topic_or_word, k=3)
        docs = retriever.invoke(f"{topic_or_word} vocabulary {level}")
    except Exception as exc:
        logger.warning("retrieve_vocab failed for %s/%s: %s", language, topic_or_word, exc)
        return f"(retrieval temporarily unavailable for vocabulary topic '{topic_or_word}')"

    if not docs:
        return "(no retrieved vocabulary available for this topic)"

    results = []
    for doc in docs:
        title = doc.metadata.get("title", doc.metadata.get("topic", "Vocabulary"))
        results.append(f"**{title}**\n{doc.page_content}")

    return "\n\n---\n\n".join(results)


@tool
def generate_exercise(language: str, level: str, skill: str) -> str:
    """Generate a structured language exercise from retrieved knowledge base content.

    Args:
        language: Target language code — "en", "ko", or "ja"
        level: Learner level — "beginner", "intermediate", or "advanced"
        skill: Skill to practice — "grammar", "vocabulary", "reading", or "writing"

    Returns:
        A structured exercise with instructions, a question, and expected answer format.
    """
    try:
        # Search without level filter to get more content across levels,
        # and use skill/topic keywords that match actual seed data.
        target_topics = {
            "grammar": "tenses grammar sentence structure clauses conditionals modals",
            "vocabulary": "food daily_routine greetings travel emotions adjectives idioms",
            "reading": "reading comprehension passage",
            "writing": "writing composition paragraph",
        }
        query = target_topics.get(skill, f"{skill}")
        retriever = _get_retriever(language, k=5)
        docs = retriever.invoke(f"{query}")
    except Exception as exc:
        logger.warning("generate_exercise retrieval failed for %s/%s: %s", language, skill, exc)
        return (
            f"Knowledge base retrieval is temporarily unavailable. "
            f"Please create a {skill} exercise for {language} at {level} level "
            f"based on your general knowledge."
        )

    if not docs:
        return (
            f"No content available to generate a {skill} exercise for {language} at {level} level. "
            f"Please create an exercise based on your general knowledge of {language} at {level} level."
        )

    context = "\n\n".join(
        f"{doc.metadata.get('topic', 'Note')} (level: {doc.metadata.get('level', 'unknown')}): {doc.page_content[:400]}..."
        for doc in docs[:4]
    )

    return (
        f"--- Exercise Context (retrieved from knowledge base) ---\n"
        f"Target level: {level}\n"
        f"{context}\n"
        f"---\n"
        f"Use the context above to create a {skill} exercise at {level} level "
        f"for a student learning {language}. If the retrieved content is at a different level, "
        f"adapt it to the target level. Include:\n"
        f"1. Clear instructions\n"
        f"2. The exercise question or prompt\n"
        f"3. The expected answer format (do NOT give the answer itself — the student will submit it)"
    )


# ---------------------------------------------------------------------------
# Week 2 tools: grade_answer + log_mistake
# ---------------------------------------------------------------------------


@tool
def grade_answer(exercise_context: str, user_answer: str, language: str, level: str) -> str:
    """Grade a student's exercise answer and provide feedback.

    Args:
        exercise_context: The exercise question/prompt that was given to the student.
        user_answer: The student's submitted answer text.
        language: Target language code — "en", "ko", or "ja".
        level: Learner level — "beginner", "intermediate", or "advanced".

    Returns:
        A JSON string with keys: correct (bool), explanation (str), correct_answer (str|None).
    """
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite",
            temperature=0,
            google_api_key=os.getenv("GEMINI_API_KEY"),
            request_timeout=30,
        )

        grading_prompt = f"""You are a strict but fair language tutor grading a student's exercise answer.

Exercise context: {exercise_context}
Student's answer: {user_answer}
Language: {language}
Level: {level}

Grade the answer. Return ONLY valid JSON with these keys:
- "correct": boolean (true if the answer is fully correct, false otherwise)
- "explanation": string (brief, encouraging explanation — tell the student what they got right and, if wrong, what the issue is)
- "correct_answer": string or null (if the answer was wrong, provide the correct answer; if correct, set to null)

Be strict on accuracy but encouraging in tone. For {language} at {level} level, adjust your strictness:
- Beginner: small mistakes are okay if the meaning is clear
- Intermediate: expect grammatically correct answers
- Advanced: expect nuanced, precise answers"""

        response = llm.invoke(grading_prompt)
        return _extract_text(response.content)
    except Exception as exc:
        logger.warning("grade_answer LLM call failed: %s", exc)
        # Return a structured fallback the agent can read
        return json.dumps({
            "correct": None,
            "explanation": "I'm having trouble grading your answer right now. Please try submitting it again in a moment.",
            "correct_answer": None,
            "error": True,
        })


@tool
def log_mistake(mistake_type: str, detail: str) -> str:
    """Log a corrected mistake to the student's session for future personalization.

    Call this whenever the student makes a mistake that you correct. This helps
    the tutor remember and reinforce weak areas in future exercises.

    Args:
        mistake_type: Category of mistake — "grammar", "vocabulary", "pronunciation", or "spelling".
        detail: Brief description of the mistake and correction (e.g. "Used 'は' instead of 'が' for subject marker").

    Returns:
        Confirmation that the mistake was logged.
    """
    session_id = _CURRENT_SESSION_ID.get()
    if not session_id:
        return "No active session — mistake not logged."

    try:
        # Lazy import to avoid circular dependency
        from .sessions import add_mistake
        add_mistake(session_id, mistake_type, detail)
        return f"Mistake logged: [{mistake_type}] {detail}"
    except Exception as exc:
        logger.warning("log_mistake failed for session %s: %s", session_id, exc)
        return f"Mistake logging failed (non-critical): {exc}"


def set_session_context(user_id: str, session_id: str) -> None:
    """Set the session context for the current request thread."""
    _CURRENT_USER_ID.set(user_id)
    _CURRENT_SESSION_ID.set(session_id)


def clear_session_context() -> None:
    """Clear the session context."""
    _CURRENT_USER_ID.set("")
    _CURRENT_SESSION_ID.set("")