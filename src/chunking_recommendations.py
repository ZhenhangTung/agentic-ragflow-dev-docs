"""
Helpers for adding general chunking-improvement library recommendations.
"""

CHUNKING_QUALITY_KEYWORDS = (
    "chunking quality",
    "improve chunking",
    "better chunking",
    "chunk quality",
    "semantic chunk",
    "semantic split",
    "chunk strategy",
    "chunking strategy",
    "machine learning lib",
    "ml lib",
)

KNOWN_ML_LIBS = ("sentence-transformers", "spacy", "nltk")

CHUNKING_ML_RECOMMENDATION = """

### External General ML/NLP Libraries (for chunking quality improvements)

If you want to improve chunking quality beyond the current built-in strategy, common options include:

- **sentence-transformers**: embedding-based semantic chunking/splitting.
- **spaCy**: robust sentence/linguistic boundary detection and rule-based preprocessing.
- **NLTK**: lightweight sentence tokenization and text segmentation utilities.

These are general recommendations and are not part of the official RAGFlow API documentation.
""".rstrip()


def should_add_chunking_recommendation(question: str) -> bool:
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in CHUNKING_QUALITY_KEYWORDS)


def answer_has_ml_library_suggestions(answer: str) -> bool:
    answer_lower = answer.lower()
    return any(lib in answer_lower for lib in KNOWN_ML_LIBS)


def append_chunking_recommendation_if_needed(question: str, answer: str) -> str:
    if not should_add_chunking_recommendation(question):
        return answer
    if answer_has_ml_library_suggestions(answer):
        return answer
    return f"{answer.rstrip()}\n\n{CHUNKING_ML_RECOMMENDATION}\n"
