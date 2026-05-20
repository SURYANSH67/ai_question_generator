import json
from typing import List, Dict, Any, Optional
from openai import OpenAI

class GenerationEngine:
    def __init__(self, api_key: str, base_url: Optional[str] = None):
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        
    def generate_question_paper(
        self,
        subject: str,
        topics: List[str],
        difficulty: str,
        bloom_levels: List[str],
        question_specs: List[Dict[str, Any]],  # List of dicts with keys: type, count, marks
        retrieved_context: List[Dict[str, Any]],
        custom_instructions: Optional[str] = None,
        model_name: str = "gpt-4o-mini"
    ) -> Dict[str, Any]:
        """
        Generates a question paper and its answer key/explanation using LLM + RAG context.
        Returns a structured dictionary (JSON-like).
        """
        # Formulate context text
        context_text = "\n\n".join([
            f"--- Source (Page {chunk['metadata'].get('page_number', 'N/A')}): ---\n{chunk['text']}"
            for chunk in retrieved_context
        ])
        
        # Format Bloom's levels and topics for the prompt
        bloom_str = ", ".join(bloom_levels) if bloom_levels else "Any"
        topics_str = ", ".join(topics) if topics else "General content from document"
        
        # Calculate total marks and structure specification string
        total_marks = 0
        specs_str_list = []
        for spec in question_specs:
            count = spec.get("count", 0)
            marks = spec.get("marks", 0)
            q_type = spec.get("type", "")
            total_marks += count * marks
            specs_str_list.append(f"- {count}x {q_type} questions ({marks} marks each)")
        specs_str = "\n".join(specs_str_list)
        
        # System instructions
        system_prompt = """You are an expert academic evaluator and question paper setter.
Your job is to generate a high-quality, professional, curriculum-compliant question paper and detailed answer key.
You MUST output your response in valid JSON format. The JSON structure MUST exactly match the format requested.
Do not wrap your output in markdown code blocks like ```json ... ```, just output the pure JSON string.
Ensure that the questions generated are strictly based on the provided reference context and follow the specified parameters.
"""

        # User instructions with template
        user_prompt = f"""
        Generate a question paper based on the following details and retrieved educational context.
        
        ### QUESTION PAPER PARAMETERS:
        - Subject: {subject}
        - Topics/Chapters: {topics_str}
        - Difficulty Level: {difficulty} (strictly tailor the vocabulary, complexity, and problem solving level to this difficulty)
        - Bloom's Taxonomy Levels: {bloom_str}
        - Total Marks: {total_marks}
        - Structure:
        {specs_str}
        
        {f"- Custom Instructions: {custom_instructions}" if custom_instructions else ""}
        
        ### RETRIEVED CONTEXT (Use this as the source of truth for the facts, concepts, and formulas):
        {context_text}
        
        ---
        
        ### JSON OUTPUT SCHEMA REQUIREMENT:
        You must generate a single JSON object with the following fields:
        {{
          "title": "A suitable descriptive title for the question paper",
          "subject": "{subject}",
          "difficulty": "{difficulty}",
          "bloom_levels": {json.dumps(bloom_levels)},
          "total_marks": {total_marks},
          "instructions": [
            "A list of 3-5 standard exam instructions for the student",
            "Example: Read all questions carefully.",
            "Example: Draw diagrams where necessary."
          ],
          "questions": [
            {{
              "id": 1,
              "type": "MCQ" or "Short Answer" or "Long Answer" or "True/False" or "Case Study",
              "marks": 1,
              "bloom_level": "the primary Bloom's level targeted",
              "text": "The text of the question. For MCQs, this is the question text.",
              "options": ["Option A", "Option B", "Option C", "Option D"],  // Include ONLY for MCQ, omit or set to null for others
              "answer": "The correct option text for MCQ, or the model detailed answer for other question types.",
              "explanation": "Step-by-step reasoning/calculation showing why this is the correct answer and which educational concept it relates to."
            }}
          ]
        }}
        
        Make sure:
        1. All questions must be relevant to the topics and derived from the retrieved context.
        2. MCQs must have exactly 4 logical options, with one clear correct answer.
        3. Answers and explanations must be mathematically/conceptually rigorous and pedagogical.
        4. Do not include markdown code block characters around the JSON output.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,
                response_format={"type": "json_object"}
            )
            raw_content = response.choices[0].message.content.strip()
            paper_data = json.loads(raw_content)
            return paper_data
        except json.JSONDecodeError as je:
            print(f"JSON Decode Error in model output: {je}")
            print(f"Raw output was: {raw_content}")
            # Try to fix or return clean error structure
            return {"error": "Failed to parse generated output as JSON. Please try again.", "raw": raw_content}
        except Exception as e:
            print(f"Error during LLM question generation: {e}")
            return {"error": f"LLM generation failed: {str(e)}"}
