from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class Dataset:
    train_tokens: List[str]
    val_tokens: List[str]
    test_tokens: List[str]
    tokens_to_id: Dict[str, int] = field(default_factory=dict)
    id_to_tokens: Dict[int, str] = field(default_factory=dict)
