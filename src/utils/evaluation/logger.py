import json
import os
from typing import List, Dict
from datetime import datetime

class QALogger:
    def __init__(self, path: str = "src/utils/evaluation/ground_truth_ragas.json", auto_timestamp: bool = False):
        """
        :param path: File path to save the log JSON
        :param auto_timestamp: If True, appends a timestamp to the filename
        """
        if auto_timestamp:
            base, ext = os.path.splitext(path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"{base}_{timestamp}{ext}"

        self.path = path
        self.data: List[Dict] = []

    def log(self, question: str, generated_answer: str, ground_truth: str, contexts: List[str]):
        """
        Add one QA interaction to the log.
        """
        self.data.append({
            "question": question,
            "generated_answer": generated_answer,
            "ground_truth": ground_truth,
            "contexts": contexts
        })

    def save(self):
        """
        Write the accumulated logs to disk as pretty-printed JSON.
        """
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
