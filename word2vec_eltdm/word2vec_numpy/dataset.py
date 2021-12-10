from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class Dataset:
    tokens: List[str]
    tokens_to_id: Dict[str, int] = field(default_factory=dict)
    id_to_tokens: Dict[int, str] = field(default_factory=dict)


@dataclass
class Text8Dataset:
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset
