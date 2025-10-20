from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
import pandas as pd
from src.ml.base_model import BaseModel

@dataclass
class ModelRecord:
    """Container for one model's data, trained model, and evaluation."""
    name: str
    data: List[pd.DataFrame]
    model: Optional[BaseModel] = None
    evaluation: Optional[Dict[str, Any]] = field(default_factory=dict)