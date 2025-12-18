import json
from pathlib import Path
from typing import List, Dict, Any, Optional

class SafeEditDataset:
    def __init__(self, data_dir: str, model_name: Optional[str] = None, size: Optional[int] = None,
                 file_path: str = "/home/ubuntu/ye/AnyEdit/data/SafeEdit_test_llama.json"):
        self.items: List[Dict[str, Any]] = []
        self._load(file_path)
        if size is not None:
            self.items = self.items[:size]

    def _load(self, file_path: str):
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"SafeEditDataset file not found: {file_path}")
        # Try standard JSON first
        try:
            with open(path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            self._from_iterable(raw)
            return
        except json.JSONDecodeError:
            pass
        # Fallback: JSON Lines
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._from_iterable([obj])

    def _from_iterable(self, iterable):
        for obj in iterable:

            general_prompt=[
                obj["generalization test"]["test input of only harmful question"],
                obj["generalization test"]["test input of other attack prompt input"],
                obj["generalization test"]["test input of other question input"],
                obj["generalization test"]["test input of other questions and attack prompts"]
            ]
            
            item = {
                "id": obj.get("id"),
                "question": obj.get("adversarial prompt"),
                "answer": obj.get("safe generation"),
                "behaviour": obj.get("question"),
                "general prompt": general_prompt,
                "knowledge constrain": obj.get("knowledge constrain"),
                "safe generation": obj.get("safe generation")
            }
            self.items.append(item)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]
