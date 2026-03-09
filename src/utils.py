import re
import datasets
from typing import Any, Dict

def to_snake_case(name: str) -> str:
    """Convert PascalCase or camelCase string to snake_case."""
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def save_signature_to_file(result: Dict[str, Any], output_path: str | None = None) -> str:
    """Save generated signature code to a Python file"""
    if not output_path:
        signature_name = result.get("signature_name", "generated_signature")
        output_path = f"{to_snake_case(signature_name)}.py"

    if result.get("code"):
        with open(output_path, "w") as f:
            f.write(result["code"])
        return output_path
    else:
        raise ValueError("No code to save")


def balanced_sample(dataset, num_samples_per_class, seed):
    unique_labels = list(set(dataset['label']))
    sets = []
    for label in unique_labels:
        _examples = dataset.filter(lambda x: x['label'] == label)
        _examples = _examples.shuffle(seed = seed).select(range(min(len(_examples), num_samples_per_class)))
        sets.append(_examples)

    balanced_dataset = datasets.concatenate_datasets(sets).shuffle(seed = seed)
    return balanced_dataset
