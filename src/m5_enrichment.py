"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.

    Args:
        text: Raw chunk text.

    Returns:
        Summary string (2-3 câu).
    """
    if not text or not text.strip():
        return ""

    if OPENAI_API_KEY:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt.",
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            pass

    sentences = [s.strip() for s in text.split(". ") if s.strip()]
    if not sentences:
        return text.strip()
    summary = ". ".join(sentences[:2])
    if not summary.endswith("."):
        summary += "."
    return summary


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).

    Args:
        text: Raw chunk text.
        n_questions: Số câu hỏi cần generate.

    Returns:
        List of question strings.
    """
    if not text or not text.strip() or n_questions <= 0:
        return []

    if OPENAI_API_KEY:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi bằng tiếng Việt "
                            "mà đoạn văn có thể trả lời. Trả về mỗi câu hỏi trên 1 dòng, "
                            "không đánh số, kết thúc bằng dấu '?'."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=200,
                temperature=0.3,
            )
            raw = resp.choices[0].message.content.strip().split("\n")
            questions = [q.strip().lstrip("0123456789.-) ").strip() for q in raw]
            questions = [q for q in questions if q]
            return questions[:n_questions]
        except Exception:
            pass

    return []


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).

    Args:
        text: Raw chunk text.
        document_title: Tên document gốc.

    Returns:
        Text với context prepended.
    """
    if not text or not text.strip():
        return text

    context = ""
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Viết 1 câu ngắn bằng tiếng Việt mô tả đoạn văn này "
                            "nằm ở đâu trong tài liệu và nói về chủ đề gì. "
                            "Chỉ trả về đúng 1 câu."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}",
                    },
                ],
                max_tokens=80,
                temperature=0.2,
            )
            context = resp.choices[0].message.content.strip()
        except Exception:
            context = ""

    if not context:
        if document_title:
            context = f"Trích từ tài liệu '{document_title}'."
        else:
            context = "Trích từ tài liệu nội bộ."

    return f"{context}\n\n{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.

    Args:
        text: Raw chunk text.

    Returns:
        Dict with extracted metadata fields.
    """
    # TODO: Implement auto metadata extraction
    # 1. from openai import OpenAI
    #    import json
    #    client = OpenAI()
    # 2. resp = client.chat.completions.create(
    #        model="gpt-4o-mini",
    #        messages=[
    #            {"role": "system", "content": 'Trích xuất metadata từ đoạn văn. Trả về JSON: {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}'},
    #            {"role": "user", "content": text},
    #        ],
    #        max_tokens=150,
    #    )
    # 3. return json.loads(resp.choices[0].message.content)
    #
    # Metadata này gắn vào chunk → enable rich filtering khi search
    # VD: filter category="policy" + topic="nghỉ phép" → precision tăng
    return {}


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks.

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: List of methods to apply. Default: ["contextual", "hyqa", "metadata"]
                 Options: "summary", "hyqa", "contextual", "metadata", "full"

    Returns:
        List of EnrichedChunk objects.
    """
    if methods is None:
        methods = ["contextual", "hyqa", "metadata"]

    enriched = []

    # TODO: Implement enrichment pipeline
    # For each chunk:
    #   1. summary = summarize_chunk(chunk["text"]) if "summary" in methods or "full" in methods
    #   2. questions = generate_hypothesis_questions(chunk["text"]) if "hyqa" in methods or "full" in methods
    #   3. enriched_text = contextual_prepend(chunk["text"], chunk["metadata"].get("source", ""))
    #      if "contextual" in methods or "full" in methods
    #   4. auto_meta = extract_metadata(chunk["text"]) if "metadata" in methods or "full" in methods
    #   5. Create EnrichedChunk(
    #          original_text=chunk["text"],
    #          enriched_text=enriched_text or chunk["text"],
    #          summary=summary or "",
    #          hypothesis_questions=questions or [],
    #          auto_metadata={**chunk["metadata"], **auto_meta},
    #          method="+".join(methods),
    #      )
    #
    # Lưu ý: Enrichment = one-time cost (offline). Dùng model rẻ (gpt-4o-mini).
    # ROI cao vì cải thiện MỌI query sau đó.

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
