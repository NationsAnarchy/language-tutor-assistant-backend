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

from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from .logging_config import get_logger
from .text_utils import extract_text

logger = get_logger(__name__)

PRACTICE_TYPES = (
    "grammar", "vocabulary", "reading", "writing", "translation", "mistake_review",
)

# Keep the exercise contract in one place.  Reading, writing, and translation
# intentionally reuse the existing knowledge base rather than adding a new
# external source.
PRACTICE_MODE_CONFIG = {
    "grammar": {
        "query": "tenses grammar sentence structure clauses conditionals modals",
        "instruction": "Create a focused grammar exercise. Ask the learner to apply one rule and do not reveal the answer.",
    },
    "vocabulary": {
        "query": "food daily_routine greetings travel emotions adjectives idioms vocabulary",
        "instruction": "Create a vocabulary exercise that checks meaning or natural usage in context.",
    },
    "reading": {
        "query": "reading comprehension passage grammar vocabulary",
        "instruction": "Create a short reading passage followed by one clear comprehension question.",
    },
    "writing": {
        "query": "writing composition paragraph grammar vocabulary",
        "instruction": "Create a concise writing prompt with a clear length or structure expectation.",
    },
    "translation": {
        "query": "translation sentence grammar vocabulary everyday conversation",
        "instruction": "Create a translation exercise between the learner's message language and the target language; provide one sentence only.",
    },
    "mistake_review": {
        "query": "grammar vocabulary common errors corrections practice",
        "instruction": "Create a targeted review exercise using the learner's recent mistakes. Prioritize the most recent mistake patterns.",
    },
}


def practice_mode_instruction(skill: str, recent_mistakes: str = "") -> tuple[str, str]:
    """Return the retrieval query and generation instruction for a practice mode.

    Unknown skills preserve the previous graceful fallback behavior for LLM tool
    calls that may contain an unexpected value.
    """
    config = PRACTICE_MODE_CONFIG.get(skill)
    if config is None:
        return skill, f"Create a {skill} exercise."
    instruction = config["instruction"]
    if skill == "mistake_review" and recent_mistakes:
        instruction += f" Recent mistakes to address: {recent_mistakes}"
    return config["query"], instruction

# ---------------------------------------------------------------------------
# Shared LLM factory — used by both graph.py and tools
# ---------------------------------------------------------------------------

_LLM_MODEL = "gemini-3.1-flash-lite"


def make_llm(temperature: float = 0.7, timeout: int = 20) -> ChatGoogleGenerativeAI:
    """Create a ChatGoogleGenerativeAI instance with timeout for graceful degradation."""
    return ChatGoogleGenerativeAI(
        model=_LLM_MODEL,
        temperature=temperature,
        google_api_key=os.getenv("GEMINI_API_KEY"),
        request_timeout=timeout,
    )

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


def _retrieve_notes(
    *,
    language: str,
    topic: str,
    level: str,
    content_type: str,
    default_title: str,
    tool_name: str,
    no_results_label: str,
) -> str:
    """Retrieve and format grammar or vocabulary notes for a tool response."""
    try:
        retriever = _get_retriever(language, level=level, topic=topic, k=3)
        docs = retriever.invoke(f"{topic} {content_type} {level}")
    except Exception as exc:
        logger.warning("%s failed for %s/%s: %s", tool_name, language, topic, exc)
        return f"(retrieval temporarily unavailable for {content_type} topic '{topic}')"

    if not docs:
        return f"(no retrieved {no_results_label} available for this topic)"

    return "\n\n---\n\n".join(
        f"**{doc.metadata.get('title', doc.metadata.get('topic', default_title))}**\n{doc.page_content}"
        for doc in docs
    )


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
    return _retrieve_notes(
        language=language,
        topic=topic,
        level=level,
        content_type="grammar",
        default_title="Grammar Note",
        tool_name="retrieve_grammar",
        no_results_label="grammar notes",
    )


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
    return _retrieve_notes(
        language=language,
        topic=topic_or_word,
        level=level,
        content_type="vocabulary",
        default_title="Vocabulary",
        tool_name="retrieve_vocab",
        no_results_label="vocabulary",
    )


@tool
def generate_exercise(language: str, level: str, skill: str, recent_mistakes: str = "") -> str:
    """Generate a structured language exercise from retrieved knowledge base content.

    Args:
        language: Target language code — "en", "ko", or "ja"
        level: Learner level — "beginner", "intermediate", or "advanced"
        skill: Skill to practice — grammar, vocabulary, reading, writing, translation, or mistake_review.
        recent_mistakes: Optional recent mistake summary for mistake-review personalization.

    Returns:
        A structured exercise with instructions, a question, and expected answer format.
    """
    try:
        # Search without level filter to get more content across levels,
        # and use skill/topic keywords that match actual seed data.
        query, instruction = practice_mode_instruction(skill, recent_mistakes)
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
        f"adapt it to the target level. {instruction} Include:\n"
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
        llm = make_llm(temperature=0, timeout=30)

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
        return extract_text(response.content)
    except Exception as exc:
        logger.warning("grade_answer LLM call failed: %s", exc)
        # Return a structured fallback the agent can read
        return json.dumps({
            "correct": None,
            "explanation": "I'm having trouble grading your answer right now. Please try submitting it again in a moment.",
            "correct_answer": None,
            "error": True,
        })


def format_grade_feedback(grade_result: str) -> str:
    """Turn a grade tool result into safe, learner-facing feedback.

    Tool results are private implementation details.  Keeping the conversion in
    the backend prevents a raw JSON object from reaching chat history, TTS, or
    the exercise UI when a model returns the tool result verbatim.
    """
    content = grade_result.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1] if "\n" in content else ""
        if content.endswith("```"):
            content = content[:-3].strip()

    try:
        result = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return "I couldn't grade that answer reliably. Please try submitting it again."

    explanation = str(result.get("explanation") or "").strip()
    correct_answer = result.get("correct_answer")
    if result.get("correct") is True:
        return "Correct!" + (f" {explanation}" if explanation else " Nice work.")
    if result.get("correct") is False:
        feedback = "Not quite."
        if explanation:
            feedback += f" {explanation}"
        if isinstance(correct_answer, str) and correct_answer.strip():
            feedback += f"\n\n**Suggested answer**\n{correct_answer.strip()}"
        return feedback
    return explanation or "I couldn't grade that answer right now. Please try submitting it again."


@tool
def log_mistake(mistake_type: str, detail: str) -> str:
    """Log a corrected mistake to the student's session for future personalization.

    Call this whenever the student makes a mistake that you correct. This helps
    the tutor remember and reinforce weak areas in future exercises.

    Args:
        mistake_type: Use exactly one broad category: "grammar" (including syntax,
            word order, tense, conjugation, articles, prepositions, and particles),
            "vocabulary" (including word choice, usage, collocations, and idioms),
            "pronunciation" (including accent, stress, tone, and intonation), or
            "spelling" (including typos, orthography, and punctuation).
        detail: Brief description of the mistake and correction (e.g. "Used 'は' instead of 'が' for subject marker").

    Returns:
        Confirmation that the mistake was logged.
    """
    return f"Mistake recorded for this turn: [{mistake_type}] {detail}"
