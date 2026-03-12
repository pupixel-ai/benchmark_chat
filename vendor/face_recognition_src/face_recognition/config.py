from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class PipelineConfig:
    input_dir: Path
    db_path: Path
    index_path: Path
    log_dir: Optional[Path] = None
    batch_size: int = 100
    batch_retry_limit: int = 2
    batch_retry_backoff_seconds: float = 1.0
    max_side: int = 1920
    det_threshold: float = 0.60
    sim_threshold: float = 0.50
    preflight_validate: bool = True
    resume: bool = True
    providers: Tuple[str, ...] = field(default_factory=lambda: ("CPUExecutionProvider",))
    allowed_extensions: Tuple[str, ...] = field(
        default_factory=lambda: (".jpg", ".jpeg", ".png", ".heic")
    )
    model_name: str = "buffalo_l"
    det_size: Tuple[int, int] = (640, 640)
    ctx_id: int = 0

    def __post_init__(self) -> None:
        input_dir = self.input_dir.expanduser().resolve()
        db_path = self.db_path.expanduser().resolve()
        index_path = self.index_path.expanduser().resolve()
        log_dir = (
            self.log_dir.expanduser().resolve()
            if self.log_dir is not None
            else db_path.parent / "logs"
        )

        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if self.batch_retry_limit < 0:
            raise ValueError("batch_retry_limit must be zero or greater")
        if self.batch_retry_backoff_seconds < 0.0:
            raise ValueError("batch_retry_backoff_seconds must be zero or greater")
        if self.max_side <= 0:
            raise ValueError("max_side must be greater than zero")
        if not 0.0 <= self.det_threshold <= 1.0:
            raise ValueError("det_threshold must be between 0.0 and 1.0")
        if not -1.0 <= self.sim_threshold <= 1.0:
            raise ValueError("sim_threshold must be between -1.0 and 1.0")
        if not self.providers:
            raise ValueError("providers must contain at least one provider")

        object.__setattr__(self, "input_dir", input_dir)
        object.__setattr__(self, "db_path", db_path)
        object.__setattr__(self, "index_path", index_path)
        object.__setattr__(self, "log_dir", log_dir)
        object.__setattr__(
            self,
            "allowed_extensions",
            tuple(ext.lower() for ext in self.allowed_extensions),
        )
        object.__setattr__(self, "providers", tuple(self.providers))

    @classmethod
    def from_args(
        cls,
        input_dir: str,
        db_path: str,
        index_path: str,
        log_dir: Optional[str] = None,
        batch_size: int = 100,
        batch_retry_limit: int = 2,
        batch_retry_backoff_seconds: float = 1.0,
        max_side: int = 1920,
        det_threshold: float = 0.60,
        sim_threshold: float = 0.50,
        preflight_validate: bool = True,
        resume: bool = True,
        providers: Optional[Sequence[str]] = None,
        model_name: str = "buffalo_l",
    ) -> "PipelineConfig":
        return cls(
            input_dir=Path(input_dir),
            db_path=Path(db_path),
            index_path=Path(index_path),
            log_dir=Path(log_dir) if log_dir else None,
            batch_size=batch_size,
            batch_retry_limit=batch_retry_limit,
            batch_retry_backoff_seconds=batch_retry_backoff_seconds,
            max_side=max_side,
            det_threshold=det_threshold,
            sim_threshold=sim_threshold,
            preflight_validate=preflight_validate,
            resume=resume,
            providers=tuple(providers or ("CPUExecutionProvider",)),
            model_name=model_name,
        )
