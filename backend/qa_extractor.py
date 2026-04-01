"""
Q&A Extraction Module

Extracts question-answer pairs from call conversation history using GPT-4.
"""

import os
import json
from typing import List, Dict, Optional
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from credentialing_agent import DatabaseManager
from loguru import logger


class QAExtractor:
    """Extract Q&A pairs from conversation history using GPT-4."""

    def __init__(self):
        """Initialize the Q&A extractor with GPT-4."""
        self.llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.1,  # Low temperature for deterministic extraction
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        self.extraction_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a Q&A extraction expert analyzing insurance credentialing phone calls.

Given a conversation transcript and a list of questions that were asked, extract the answers that were provided by the insurance representative.

For each question:
1. Find where it was asked in the conversation
2. Identify the answer(s) provided by the representative
3. Extract the relevant answer text (be concise but complete)
4. Provide confidence (0.0-1.0) based on clarity and completeness of the answer

Return JSON array in this exact format:
[
  {{
    "question_index": 0,
    "question_text": "exact question from the list",
    "answer_text": "the representative's answer or summary of their response",
    "confidence": 0.95,
    "conversation_snippet": [
      {{"speaker": "agent", "text": "question text"}},
      {{"speaker": "representative", "text": "answer text"}}
    ]
  }}
]

Guidelines:
- If a question wasn't answered, set answer_text to null and confidence to 0.0
- Include 2-4 messages in conversation_snippet for context
- Confidence should reflect how clear and direct the answer was
- Combine multiple representative responses if needed for a complete answer
- Be factual - don't infer information that wasn't explicitly stated
""",
                ),
                (
                    "user",
                    """Questions asked during the call:
{questions}

Full conversation transcript:
{transcript}

Extract all Q&A pairs and return as JSON array:""",
                ),
            ]
        )

    def extract_qa_pairs(
        self, questions: List[str], conversation: List[Dict]
    ) -> List[Dict]:
        """
        Extract Q&A pairs from conversation.

        Args:
            questions: List of question strings that were asked
            conversation: List of conversation messages with 'speaker' and 'message' keys

        Returns:
            List of Q&A pair dictionaries
        """
        # Format conversation for prompt
        transcript = "\n".join(
            [f"{msg['speaker']}: {msg['message']}" for msg in conversation]
        )

        # Format questions with indices
        questions_formatted = "\n".join(
            [f"{i}. {q}" for i, q in enumerate(questions)]
        )

        try:
            # Extract using LangChain
            chain = self.extraction_prompt | self.llm | JsonOutputParser()
            result = chain.invoke(
                {"questions": questions_formatted, "transcript": transcript}
            )

            # Validate and ensure question_index is present
            for qa in result:
                if "question_index" not in qa:
                    # Try to match question_text to find index
                    for i, q in enumerate(questions):
                        if q.lower() in qa.get("question_text", "").lower():
                            qa["question_index"] = i
                            break

            return result

        except Exception as e:
            logger.error(f"Q&A extraction failed: {e}")
            # Return empty Q&A pairs with low confidence
            return [
                {
                    "question_index": i,
                    "question_text": q,
                    "answer_text": None,
                    "confidence": 0.0,
                    "conversation_snippet": [],
                }
                for i, q in enumerate(questions)
            ]

    def save_qa_pairs(
        self, call_id: str, request_id: str, qa_pairs: List[Dict]
    ) -> int:
        """
        Save Q&A pairs to database.

        Args:
            call_id: The call identifier
            request_id: The credentialing request ID
            qa_pairs: List of Q&A pair dictionaries

        Returns:
            Number of pairs saved
        """
        db = DatabaseManager()
        try:
            count = db.save_qa_pairs(call_id, request_id, qa_pairs)
            logger.info(f"Saved {count} Q&A pairs for call {call_id}")
            return count
        except Exception as e:
            logger.error(f"Failed to save Q&A pairs for {call_id}: {e}")
            raise
        finally:
            db.close()


def extract_qa_async(call_id: str, request_id: Optional[str] = None):
    """
    Background job to extract Q&A pairs for a completed call.

    Args:
        call_id: The call identifier to process
    """
    try:
        logger.info(f"Starting Q&A extraction for call {call_id}")
        db = DatabaseManager()

        # Get call data - questions and conversation
        with db.conn.cursor() as cur:
            # First, get the request data with questions
            cur.execute(
                """
                SELECT DISTINCT
                    ch.request_id,
                    r.questions
                FROM conversation_history ch
                JOIN credentialing_requests r ON r.id = ch.request_id
                WHERE ch.call_id = %s
                   OR (%s IS NOT NULL AND ch.request_id = %s)
                LIMIT 1
                """,
                (call_id, request_id, request_id),
            )

            row = cur.fetchone()
            if not row:
                logger.warning(
                    f"No conversation found for call {call_id}"
                    + (f" (request_id={request_id})" if request_id else "")
                )
                db.close()
                return

            request_id = str(row[0])
            questions = row[1]  # JSONB array

            # Get full conversation history
            cur.execute(
                """
                SELECT speaker, message, timestamp
                FROM conversation_history
                WHERE call_id = %s
                   OR (%s IS NOT NULL AND request_id = %s)
                ORDER BY timestamp ASC
                """,
                (call_id, request_id, request_id),
            )

            conversation = [
                {"speaker": row[0], "message": row[1], "timestamp": row[2]}
                for row in cur.fetchall()
            ]

        db.close()

        if not questions or not conversation:
            logger.warning(
                f"Missing questions or conversation for call {call_id}"
            )
            return

        # Extract Q&A pairs
        extractor = QAExtractor()
        qa_pairs = extractor.extract_qa_pairs(questions, conversation)

        # Save to database
        extractor.save_qa_pairs(call_id, request_id, qa_pairs)

        logger.info(f"Successfully extracted {len(qa_pairs)} Q&A pairs for call {call_id}")

    except Exception as e:
        logger.error(f"Q&A extraction failed for call {call_id}: {e}", exc_info=True)


if __name__ == "__main__":
    # Test with a sample call_id
    import sys

    if len(sys.argv) > 1:
        call_id = sys.argv[1]
        extract_qa_async(call_id)
    else:
        print("Usage: python qa_extractor.py <call_id>")
