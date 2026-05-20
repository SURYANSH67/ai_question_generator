import json
from typing import List, Dict, Any
from openai import OpenAI

class AnswerEvaluator:
    def __init__(self, api_key: str, base_url: str = None):
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        
    def evaluate_answers(
        self,
        question_paper: Dict[str, Any],
        student_answers: Dict[str, str], # Maps question_id (str/int) to student's answer text
        model_name: str = "gpt-4o-mini"
    ) -> Dict[str, Any]:
        """
        Evaluates a student's answer sheet against a generated question paper's answers.
        Returns a detailed report including grades, semantic analysis, and qualitative feedback.
        """
        questions = question_paper.get("questions", [])
        evaluation_items = []
        
        for q in questions:
            q_id = q.get("id")
            q_type = q.get("type", "SA")
            max_marks = q.get("marks", 1)
            q_text = q.get("text", "")
            correct_answer = q.get("answer", "")
            explanation = q.get("explanation", "")
            
            student_ans = student_answers.get(str(q_id), "").strip()
            
            if not student_ans:
                # Unanswered question
                evaluation_items.append({
                    "id": q_id,
                    "text": q_text,
                    "type": q_type,
                    "max_marks": max_marks,
                    "student_answer": "",
                    "score_awarded": 0.0,
                    "semantic_similarity": 0.0,
                    "key_concepts_covered": [],
                    "key_concepts_missing": ["Entire answer is missing"],
                    "feedback": "No answer provided. Zero marks awarded.",
                    "status": "Unanswered"
                })
                continue
                
            # For MCQ or True/False, we can do direct comparison or quick LLM check
            if q_type in ["MCQ", "True/False"]:
                is_correct = False
                # Direct string equality or case-insensitive matching
                if student_ans.lower() == correct_answer.lower():
                    is_correct = True
                # Sometimes MCQs are answered as "A" or "Option A"
                elif len(student_ans) == 1 and hasattr(q, "options") and q.get("options"):
                    idx = ord(student_ans.upper()) - 65 # 'A' -> 0, 'B' -> 1
                    options = q.get("options")
                    if 0 <= idx < len(options) and options[idx].lower() == correct_answer.lower():
                        is_correct = True
                
                score = float(max_marks) if is_correct else 0.0
                feedback = "Correct answer!" if is_correct else f"Incorrect. The correct answer is: {correct_answer}."
                evaluation_items.append({
                    "id": q_id,
                    "text": q_text,
                    "type": q_type,
                    "max_marks": max_marks,
                    "student_answer": student_ans,
                    "score_awarded": score,
                    "semantic_similarity": 1.0 if is_correct else 0.0,
                    "key_concepts_covered": [correct_answer] if is_correct else [],
                    "key_concepts_missing": [] if is_correct else [correct_answer],
                    "feedback": feedback + f" {explanation}",
                    "status": "Correct" if is_correct else "Incorrect"
                })
            else:
                # Use LLM to evaluate Short Answer, Long Answer, and Case Study
                eval_prompt = f"""
                You are an academic grader evaluating a student's answer.
                Compare the student's answer with the correct answer key and explanation.
                Award a score out of {max_marks} marks based on the accuracy, completeness, and correctness of the answer.
                
                Question:
                "{q_text}"
                
                Expected Answer / Criteria:
                "{correct_answer}"
                
                Expected Explanation / Concept:
                "{explanation}"
                
                Student's Answer:
                "{student_ans}"
                
                Provide the evaluation as a JSON object with the following fields:
                {{
                  "score_awarded": (float between 0.0 and {max_marks}),
                  "semantic_similarity": (float between 0.0 and 1.0 indicating how close the meaning is to the expected answer),
                  "key_concepts_covered": ["concept1", "concept2"],
                  "key_concepts_missing": ["missing_concept1"],
                  "feedback": "constructive feedback telling the student what is correct, what is missing, and how to improve. Be professional and encouraging."
                }}
                Do not include any markdown styling or extra characters, return ONLY the JSON object.
                """
                
                try:
                    response = self.client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": "You are a professional teacher grading student answers. You must output JSON."},
                            {"role": "user", "content": eval_prompt}
                        ],
                        temperature=0.2,
                        response_format={"type": "json_object"}
                    )
                    eval_data = json.loads(response.choices[0].message.content.strip())
                    
                    # Ensure score doesn't exceed maximum marks
                    score_awarded = min(float(eval_data.get("score_awarded", 0.0)), float(max_marks))
                    
                    status = "Incorrect"
                    if score_awarded == max_marks:
                        status = "Correct"
                    elif score_awarded > 0:
                        status = "Partial"
                        
                    evaluation_items.append({
                        "id": q_id,
                        "text": q_text,
                        "type": q_type,
                        "max_marks": max_marks,
                        "student_answer": student_ans,
                        "score_awarded": score_awarded,
                        "semantic_similarity": float(eval_data.get("semantic_similarity", 0.0)),
                        "key_concepts_covered": eval_data.get("key_concepts_covered", []),
                        "key_concepts_missing": eval_data.get("key_concepts_missing", []),
                        "feedback": eval_data.get("feedback", ""),
                        "status": status
                    })
                except Exception as e:
                    print(f"Error grading question {q_id}: {e}")
                    # Safe fallback
                    evaluation_items.append({
                        "id": q_id,
                        "text": q_text,
                        "type": q_type,
                        "max_marks": max_marks,
                        "student_answer": student_ans,
                        "score_awarded": 0.0,
                        "semantic_similarity": 0.0,
                        "key_concepts_covered": [],
                        "key_concepts_missing": [],
                        "feedback": f"Failed to grade answer automatically: {str(e)}",
                        "status": "Ungraded"
                    })
                    
        # Compute overall stats
        total_max_marks = float(sum(item["max_marks"] for item in evaluation_items))
        total_score_awarded = float(sum(item["score_awarded"] for item in evaluation_items))
        percentage = (total_score_awarded / total_max_marks * 100) if total_max_marks > 0 else 0.0
        
        grade = "F"
        if percentage >= 90: grade = "A+"
        elif percentage >= 80: grade = "A"
        elif percentage >= 70: grade = "B"
        elif percentage >= 60: grade = "C"
        elif percentage >= 50: grade = "D"
        
        evaluation_report = {
            "title": question_paper.get("title", "Question Paper Evaluation"),
            "subject": question_paper.get("subject", "N/A"),
            "difficulty": question_paper.get("difficulty", "Medium"),
            "total_questions": len(questions),
            "total_max_marks": total_max_marks,
            "total_score_awarded": round(total_score_awarded, 2),
            "percentage": round(percentage, 2),
            "grade": grade,
            "items": evaluation_items
        }
        
        return evaluation_report
        
    def generate_mock_student_answers(self, question_paper: Dict[str, Any], accuracy: str = "good") -> Dict[str, str]:
        """
        Generates simulated student answers for demonstration/testing.
        accuracy can be: 'good' (high quality answers), 'medium' (partially correct), 'poor' (incorrect/missing).
        """
        questions = question_paper.get("questions", [])
        mock_answers = {}
        
        for q in questions:
            q_id = str(q.get("id"))
            q_type = q.get("type", "MCQ")
            correct_ans = q.get("answer", "")
            
            if q_type == "MCQ":
                if accuracy == "good":
                    mock_answers[q_id] = correct_ans
                elif accuracy == "medium":
                    # 60% chance correct, otherwise pick another option if available
                    import random
                    options = q.get("options", [])
                    if options and random.random() > 0.6:
                        wrong_options = [o for o in options if o != correct_ans]
                        mock_answers[q_id] = random.choice(wrong_options) if wrong_options else correct_ans
                    else:
                        mock_answers[q_id] = correct_ans
                else:
                    # pick a wrong option
                    options = q.get("options", [])
                    wrong_options = [o for o in options if o != correct_ans]
                    mock_answers[q_id] = random.choice(wrong_options) if wrong_options else "I don't know"
            elif q_type == "True/False":
                if accuracy == "good":
                    mock_answers[q_id] = correct_ans
                elif accuracy == "medium":
                    mock_answers[q_id] = correct_ans if hash(q_id) % 2 == 0 else ("True" if correct_ans == "False" else "False")
                else:
                    mock_answers[q_id] = "True" if correct_ans == "False" else "False"
            else:
                # Open ended questions (Short/Long Answer)
                if accuracy == "good":
                    # Close paraphrasing of correct answer
                    mock_answers[q_id] = f"I think that {correct_ans[0].lower() + correct_ans[1:]} This explanation holds because it directly corresponds to the source facts."
                elif accuracy == "medium":
                    # Partial explanation
                    half_len = len(correct_ans) // 2
                    mock_answers[q_id] = correct_ans[:half_len] + "... which explains the phenomenon partly."
                else:
                    # Incorrect/Irrelevant answer
                    mock_answers[q_id] = "This is a very difficult topic. I believe this relates to something else entirely, maybe general scientific concepts like energy conservation."
                    
        return mock_answers
