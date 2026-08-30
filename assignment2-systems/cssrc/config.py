from dataclasses import dataclass


@dataclass
class ModelConfig:
    label: str
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int

Context_length = 512

SIZES = {
    "small": ModelConfig("small", 768, 3072, 12, 12),
    "medium": ModelConfig("medium", 1024, 4096, 24, 16),
    "large": ModelConfig("large", 1280, 5120, 36, 20),
    "xl": ModelConfig("xl", 2560, 10240, 32, 32),
    "10B": ModelConfig("10B", 4608, 12288, 50, 36),
}

