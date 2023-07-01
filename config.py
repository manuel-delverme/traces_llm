import dataclasses
from dataclasses import dataclass
from typing import Dict, Any

from hydra.core.config_store import ConfigStore


@dataclass
class ModelConfig:
    _target_: str = "text_only_baseline.GPT2FineTuning"
    token_context_len: int = 8
    points_in_motor_sequence: int = 64
    max_chars_per_token: int = 20
    learning_rate: float = 3e-5
    num_layers: int = 1
    hidden_size: int = 256


@dataclasses.dataclass
class SlurmConf:
    _target_: str = "hydra_plugins.hydra_submitit_launcher.conf.SubmititSlurmConf"
    nodes: int = 1
    gpus_per_node: int = 1
    ntasks_per_node: int = 1
    time: int = 60  # in minutes
    partition: str = "main"


@dataclasses.dataclass
class Config:
    defaults = [
        {"hydra/launcher": "submitit_slurm"}
    ]
    model_config: ModelConfig = ModelConfig()
    hydra: Dict[str, Any] = dataclasses.field(default_factory=lambda: {"launcher": SlurmConf()})


cs = ConfigStore.instance()
# Register the Config dataclass with Hydra
cs.store(name="myconfig", node=Config)
