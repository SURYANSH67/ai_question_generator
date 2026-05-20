import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

class HistoryManager:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.papers_file = os.path.join(data_dir, "papers_history.json")
        self.evaluations_file = os.path.join(data_dir, "evaluations_history.json")
        
        # Initialize files if they don't exist
        if not os.path.exists(self.papers_file):
            self._save_json(self.papers_file, [])
        if not os.path.exists(self.evaluations_file):
            self._save_json(self.evaluations_file, [])

    def _load_json(self, file_path: str) -> List[Dict[str, Any]]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []

    def _save_json(self, file_path: str, data: List[Dict[str, Any]]) -> None:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving {file_path}: {e}")

    def save_paper(self, paper_data: Dict[str, Any]) -> str:
        """
        Saves a generated question paper and adds metadata.
        Returns the unique paper ID.
        """
        papers = self._load_json(self.papers_file)
        
        # Add id and timestamp if not present
        paper_id = paper_data.get("paper_id", str(uuid.uuid4())[:8])
        paper_data["paper_id"] = paper_id
        if "created_at" not in paper_data:
            paper_data["created_at"] = datetime.now().isoformat()
            
        # Avoid duplicates by matching id
        papers = [p for p in papers if p.get("paper_id") != paper_id]
        papers.append(paper_data)
        
        self._save_json(self.papers_file, papers)
        return paper_id

    def load_papers(self) -> List[Dict[str, Any]]:
        """Loads all historical papers, sorted by creation time (newest first)."""
        papers = self._load_json(self.papers_file)
        return sorted(papers, key=lambda x: x.get("created_at", ""), reverse=True)

    def get_paper(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a specific paper by its ID."""
        papers = self.load_papers()
        for p in papers:
            if p.get("paper_id") == paper_id:
                return p
        return None

    def delete_paper(self, paper_id: str) -> bool:
        """Deletes a paper and all its associated evaluations."""
        papers = self._load_json(self.papers_file)
        new_papers = [p for p in papers if p.get("paper_id") != paper_id]
        self._save_json(self.papers_file, new_papers)
        
        # Also clean up evaluations for this paper
        evaluations = self._load_json(self.evaluations_file)
        new_evaluations = [e for e in evaluations if e.get("paper_id") != paper_id]
        self._save_json(self.evaluations_file, new_evaluations)
        
        return len(papers) != len(new_papers)

    def save_evaluation(self, eval_report: Dict[str, Any], paper_id: str, student_name: str) -> str:
        """
        Saves an evaluation report for a specific paper and student.
        """
        evaluations = self._load_json(self.evaluations_file)
        eval_id = str(uuid.uuid4())[:8]
        
        eval_record = {
            "eval_id": eval_id,
            "paper_id": paper_id,
            "student_name": student_name,
            "created_at": datetime.now().isoformat(),
            "score_awarded": eval_report.get("total_score_awarded"),
            "max_marks": eval_report.get("total_max_marks"),
            "percentage": eval_report.get("percentage"),
            "grade": eval_report.get("grade"),
            "report": eval_report
        }
        
        evaluations.append(eval_record)
        self._save_json(self.evaluations_file, evaluations)
        return eval_id

    def load_evaluations(self, paper_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Loads historical evaluations, optionally filtered by paper ID."""
        evaluations = self._load_json(self.evaluations_file)
        if paper_id:
            evaluations = [e for e in evaluations if e.get("paper_id") == paper_id]
        return sorted(evaluations, key=lambda x: x.get("created_at", ""), reverse=True)

    def get_dashboard_stats(self) -> Dict[str, Any]:
        """
        Calculates key analytics and statistics for the dashboard.
        """
        papers = self.load_papers()
        evaluations = self.load_evaluations()
        
        if not papers:
            return {
                "total_papers": 0,
                "total_evaluations": 0,
                "average_score": 0.0,
                "difficulty_distribution": {},
                "subject_distribution": {},
                "score_by_subject": {},
                "papers_df": pd.DataFrame(),
                "evals_df": pd.DataFrame()
            }
            
        # Convert to Pandas for aggregation
        papers_data = []
        for p in papers:
            # Count questions by type
            q_types = {}
            for q in p.get("questions", []):
                t = q.get("type", "SA")
                q_types[t] = q_types.get(t, 0) + 1
                
            papers_data.append({
                "Paper ID": p.get("paper_id"),
                "Title": p.get("title"),
                "Subject": p.get("subject"),
                "Difficulty": p.get("difficulty"),
                "Total Marks": p.get("total_marks"),
                "Questions Count": len(p.get("questions", [])),
                "Created At": p.get("created_at"),
                "MCQ Count": q_types.get("MCQ", 0),
                "SA Count": q_types.get("Short Answer", 0),
                "LA Count": q_types.get("Long Answer", 0),
                "TF Count": q_types.get("True/False", 0),
                "CS Count": q_types.get("Case Study", 0)
            })
        papers_df = pd.DataFrame(papers_data)
        
        evals_data = []
        for e in evaluations:
            evals_data.append({
                "Evaluation ID": e.get("eval_id"),
                "Paper ID": e.get("paper_id"),
                "Student Name": e.get("student_name"),
                "Score": e.get("score_awarded"),
                "Max Marks": e.get("max_marks"),
                "Percentage": e.get("percentage"),
                "Grade": e.get("grade"),
                "Created At": e.get("created_at")
            })
        evals_df = pd.DataFrame(evals_data)
        
        # Merge to get subject/difficulty for evaluations
        if not evals_df.empty and not papers_df.empty:
            merged_df = pd.merge(evals_df, papers_df[["Paper ID", "Subject", "Difficulty"]], on="Paper ID", how="left")
        else:
            merged_df = evals_df
            
        # Distributions
        diff_dist = papers_df["Difficulty"].value_counts().to_dict() if "Difficulty" in papers_df.columns else {}
        subj_dist = papers_df["Subject"].value_counts().to_dict() if "Subject" in papers_df.columns else {}
        
        avg_score = 0.0
        score_by_subject = {}
        if not merged_df.empty:
            avg_score = float(merged_df["Percentage"].mean())
            if "Subject" in merged_df.columns:
                score_by_subject = merged_df.groupby("Subject")["Percentage"].mean().round(2).to_dict()
                
        return {
            "total_papers": len(papers),
            "total_evaluations": len(evaluations),
            "average_score": round(avg_score, 2),
            "difficulty_distribution": diff_dist,
            "subject_distribution": subj_dist,
            "score_by_subject": score_by_subject,
            "papers_df": papers_df,
            "evals_df": evals_df
        }
