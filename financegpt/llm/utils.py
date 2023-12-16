from dataclasses import dataclass


@dataclass
class InferenceResults:
    output: str
    error_code: int = 0
