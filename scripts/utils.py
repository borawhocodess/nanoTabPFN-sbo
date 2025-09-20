# scripts/utils.py

import json
import os
import sys
import uuid
from datetime import datetime

import sklearn
import torch


def pxp(x, *, on = True):
    if on:
        print()
        print(x)
        print()
    else:
        pass


def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_uuid4():
    return str(uuid.uuid4())


def get_target_dir(*subdirs, verbose=False):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)

    if verbose:
        pxp(f"root dir: {root_dir}")

    target_dir = os.path.join(root_dir, *subdirs)

    os.makedirs(target_dir, exist_ok=True)

    if verbose:
        pxp(f"target dir: {target_dir}")

    return target_dir


def _to_cpu(obj):
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = type(obj)
        return t(_to_cpu(v) for v in obj)
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    try:
        if hasattr(obj, "to"):
            return obj.to("cpu")
    except Exception:
        pass
    return obj


def _atomic_torch_save(obj, path: str):
    tmp_path = f"{path}.tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


class ArtifactHandler:
    def __init__(
        self,
        *,
        checkpoint_path: str | None,
        artifact_base: str | None,
        version: str,
        task_type: str,
        arch: dict | None,
        timestamp: str,
        uuid4: str,
        bucket_edges=None,
    ) -> None:
        if checkpoint_path:
            self.checkpoint_path = checkpoint_path
        else:
            checkpoints_dir = get_target_dir("other", "checkpoints")
            self.checkpoint_path = os.path.join(checkpoints_dir, "latest_checkpoint.pth")
        self.artifact_base = artifact_base
        self.version = version
        self.task_type = task_type
        self.arch = arch
        self.timestamp = timestamp
        self.uuid4 = uuid4
        self.bucket_edges = bucket_edges

        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)

        self.last_checkpoint_path: str | None = None
        self.last_artifact_path: str | None = None

    @staticmethod
    def load_checkpoint_file(path: str, map_location="cpu"):
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at {path!r}")

        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        if not (
            isinstance(ckpt, dict)
            and "resume" in ckpt
            and "arch" in ckpt
            and isinstance(ckpt.get("model"), dict)
            and "state_dict" in ckpt["model"]
        ):
            raise ValueError(f"Loaded file {path!r} is not a valid checkpoint.")
        meta = ckpt.get("meta")
        if not (isinstance(meta, dict) and meta.get("kind") == "checkpoint"):
            raise ValueError(f"Loaded file {path!r} is not a checkpoint artifact.")
        return ckpt

    def _base_obj(self, kind: str, state_dict):
        if not self.arch:
            raise ValueError("RunManager requires arch to be set before saving artifacts.")
        model_payload = {
            "state_dict": _to_cpu(state_dict),
        }
        if self.bucket_edges is not None:
            model_payload["bucket_edges"] = _to_cpu(self.bucket_edges)

        return {
            "id": {
                "timestamp": self.timestamp,
                "uuid4": self.uuid4,
            },
            "meta": {
                "version": self.version,
                "task_type": self.task_type,
                "kind": kind,
            },
            "arch": self.arch,
            "model": model_payload,
        }

    def save_checkpoint(self, model, optimizer_state: dict, epoch: int) -> str:
        obj = self._base_obj("checkpoint", model.state_dict())
        obj["resume"] = {
            "epoch": int(epoch),
            "optimizer": _to_cpu(optimizer_state),
        }
        _atomic_torch_save(obj, self.checkpoint_path)
        self.last_checkpoint_path = self.checkpoint_path
        return self.checkpoint_path

    def save_artifact(self, model) -> str:
        if not self.artifact_base:
            raise ValueError("artifact_base is not set for RunManager")
        base, ext = os.path.splitext(self.artifact_base)
        ext = ext if ext in {".pth", ".pt"} else ".pth"
        save_path = f"{base}_{self.timestamp}{ext}"

        obj = self._base_obj("artifact", model.state_dict())
        _atomic_torch_save(obj, save_path)
        self.last_artifact_path = save_path
        return save_path

    def update_bucket_edges(self, bucket_edges) -> None:
        self.bucket_edges = bucket_edges

    def update_artifact_base(self, artifact_base: str | None) -> None:
        self.artifact_base = artifact_base

    def update_arch(self, arch: dict) -> None:
        self.arch = arch


class MetadataHandler:
    def __init__(
        self,
        *,
        metadata_dir: str | None,
        version: str,
        task_type: str,
        timestamp: str,
        uuid4: str,
    ) -> None:
        self.version = version
        self.task_type = task_type
        self.timestamp = timestamp
        self.uuid4 = uuid4
        if metadata_dir:
            self.metadata_dir = metadata_dir
            os.makedirs(self.metadata_dir, exist_ok=True)
        else:
            self.metadata_dir = get_target_dir("other", "metadata")

        self.metadata_path = os.path.join(
            self.metadata_dir, f"run_metadata_{self.timestamp}.json"
        )
        self.metric_field = None
        self._metadata_entry: dict | None = None

    def init_metadata(self, args_dict: dict | None, device_str: str | None, metric_field: str | None):
        """
        Initialize or load per-run metadata.
        If a metadata file for this timestamp already exists, load and update it
        so resumed runs continue logging into the same JSON.
        """
        self.metric_field = metric_field

        existing: dict | None
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:
                existing = None
        else:
            existing = None

        existing_meta = existing.get("meta") if isinstance(existing, dict) else {}
        existing_env = existing.get("env") if isinstance(existing, dict) else {}
        existing_epochs = existing.get("epochs") if isinstance(existing, dict) else {}
        existing_args = existing.get("args") if isinstance(existing, dict) else None

        completed = False
        if isinstance(existing_meta, dict):
            completed = bool(existing_meta.get("completed", False))
        elif isinstance(existing, dict):
            completed = bool(existing.get("completed", False))

        args_data = args_dict if args_dict is not None else {}
        if not args_data and isinstance(existing_args, dict):
            args_data = existing_args

        env_base = existing_env if isinstance(existing_env, dict) else {}
        env_update = {
            "device": str(device_str) if device_str is not None else None,
            "python": sys.version.split()[0],
            "torch": getattr(torch, "__version__", None),
            "sklearn": getattr(sklearn, "__version__", None),
        }
        env_data = {**env_base, **env_update}

        epochs_data = existing_epochs if isinstance(existing_epochs, dict) else {}

        self._metadata_entry = {
            "id": {
                "timestamp": self.timestamp,
                "uuid4": self.uuid4,
            },
            "meta": {
                "version": self.version,
                "task_type": self.task_type,
                "completed": completed,
            },
            "args": args_data,
            "env": env_data,
            "epochs": epochs_data,
        }

        self._write_metadata()

    def _write_metadata(self):
        if self._metadata_entry is None:
            return
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata_entry, f, indent=2, sort_keys=False)

    def log_epoch(
        self,
        epoch: int,
        epoch_time: float,
        mean_loss: float,
        metric_field: str | None = None,
        metric_value: float | None = None,
    ):
        if self._metadata_entry is None:
            return
        if metric_value is None:
            raise ValueError("metric_value must be provided")
        if metric_field:
            self.metric_field = metric_field
        ek = str(int(epoch))
        self._metadata_entry["epochs"][ek] = {
            "epoch": int(epoch),
            "epoch_time": float(epoch_time),
            "mean_loss": float(mean_loss),
            (self.metric_field or "metric"): float(metric_value),
        }
        self._write_metadata()

    def mark_completed(self, expected_final_epoch: int):
        if self._metadata_entry is None:
            return
        recorded = {int(k) for k in self._metadata_entry.get("epochs", {}).keys()}
        completed = expected_final_epoch in recorded
        self._metadata_entry["meta"]["completed"] = completed
        self._write_metadata()


class RunManager:
    """Coordinates per-run metadata and artifact persistence."""

    VERSION = "0.0"

    def __init__(self, args, device=None) -> None:
        self.args = args
        self.device = device

        task_type = getattr(args, "type", None)
        if task_type is None:
            raise ValueError("RunManager requires 'type' to be defined on args")

        artifact_base = getattr(args, "saveweights", None)
        checkpoint_path = getattr(args, "checkpoint_path", None)
        metadata_dir = getattr(args, "metadata_dir", None)
        load_checkpoint_path = getattr(args, "loadcheckpoint", None)
        arch = getattr(args, "arch", None)
        bucket_edges = getattr(args, "bucket_edges", None)
        timestamp = getattr(args, "timestamp", None)
        uuid4 = getattr(args, "uuid4", None)

        priordump = getattr(args, "priordump", None)
        steps = getattr(args, "steps", None)
        accumulate = getattr(args, "accumulate", None)

        if hasattr(args, "priordump"):
            self._validate_priordump(priordump)
        self._validate_gradient_accumulation(steps, accumulate)
        self._ensure_artifact_directory(artifact_base)

        loaded_ckpt = None
        if load_checkpoint_path:
            loaded_ckpt = ArtifactHandler.load_checkpoint_file(load_checkpoint_path, map_location="cpu")

        self.resume_checkpoint = loaded_ckpt
        self.resuming = loaded_ckpt is not None
        self.resume_epoch = int(loaded_ckpt["resume"]["epoch"]) if loaded_ckpt else 0

        if loaded_ckpt and isinstance(loaded_ckpt.get("id"), dict):
            timestamp = loaded_ckpt["id"].get("timestamp", timestamp)
            uuid4 = loaded_ckpt["id"].get("uuid4", uuid4)

        ckpt_meta = loaded_ckpt.get("meta") if loaded_ckpt else None
        self.ckpt_meta = ckpt_meta if isinstance(ckpt_meta, dict) else {}
        ckpt_task_type = self.ckpt_meta.get("task_type") if self.ckpt_meta else None
        if ckpt_task_type and ckpt_task_type != task_type:
            raise ValueError(
                f"Checkpoint task_type {ckpt_task_type!r} does not match requested type {task_type!r}."
            )

        if loaded_ckpt and "arch" in loaded_ckpt:
            arch = loaded_ckpt["arch"]

        ckpt_bucket_edges = None
        if loaded_ckpt and isinstance(loaded_ckpt.get("model"), dict):
            ckpt_bucket_edges = loaded_ckpt["model"].get("bucket_edges")

        if ckpt_bucket_edges is not None:
            bucket_edges = self._normalize_bucket_edges(ckpt_bucket_edges)

        self.arch = arch
        self.task_type = task_type
        self.bucket_edges = self._normalize_bucket_edges(bucket_edges)
        self.artifact_base = artifact_base
        self.timestamp = timestamp or get_timestamp()
        self.uuid4 = uuid4 or get_uuid4()

        self.metadatahandler = MetadataHandler(
            metadata_dir=metadata_dir,
            version=self.VERSION,
            task_type=self.task_type,
            timestamp=self.timestamp,
            uuid4=self.uuid4,
        )
        self.metadata_dir = self.metadatahandler.metadata_dir
        self.artifacthandler = ArtifactHandler(
            checkpoint_path=checkpoint_path,
            artifact_base=self.artifact_base,
            version=self.VERSION,
            task_type=self.task_type,
            arch=self.arch,
            timestamp=self.timestamp,
            uuid4=self.uuid4,
            bucket_edges=self.bucket_edges,
        )
        self.checkpoint_path = self.artifacthandler.checkpoint_path

        if self.arch is not None:
            self.artifacthandler.update_arch(self.arch)

        self.init_metadata()

    @property
    def latest_checkpoint_path(self) -> str | None:
        if self.artifacthandler.last_checkpoint_path:
            return self.artifacthandler.last_checkpoint_path
        return self.artifacthandler.checkpoint_path

    @property
    def latest_artifact_path(self) -> str | None:
        return self.artifacthandler.last_artifact_path

    def update_bucket_edges(self, bucket_edges) -> None:
        self.bucket_edges = self._normalize_bucket_edges(bucket_edges)
        self.artifacthandler.update_bucket_edges(self.bucket_edges)

    def update_artifact_base(self, artifact_base: str | None) -> None:
        self.artifact_base = artifact_base
        self._ensure_artifact_directory(artifact_base)
        self.artifacthandler.update_artifact_base(artifact_base)

    def update_arch(self, arch: dict) -> None:
        if arch is None:
            return
        if self.arch is not None and self.arch != arch:
            raise ValueError("Provided architecture does not match checkpoint architecture.")
        self.arch = arch
        self.artifacthandler.update_arch(arch)

    def init_metadata(
        self,
        args_dict: dict | None = None,
        device_str: str | None = None,
        metric_field: str | None = None,
    ):
        if args_dict is None and hasattr(self.args, "__dict__"):
            args_dict = vars(self.args)
        if device_str is None and self.device is not None:
            device_str = str(self.device)
        self.metadatahandler.init_metadata(args_dict, device_str, metric_field)

    def log_epoch(
        self,
        epoch: int,
        epoch_time: float,
        mean_loss: float,
        metric_field: str | None = None,
        metric_value: float | None = None,
    ):
        if metric_value is None:
            raise ValueError("metric_value must be provided")
        self.metadatahandler.log_epoch(
            epoch,
            epoch_time,
            mean_loss,
            metric_field,
            metric_value,
        )

    def _validate_priordump(self, priordump: str | None) -> None:
        if not priordump or not os.path.isfile(priordump):
            raise FileNotFoundError(f"Prior dump not found at {priordump!r}")

    def _validate_gradient_accumulation(self, steps, accumulate) -> None:
        if steps is None or accumulate is None:
            return
        try:
            divisible = steps % accumulate == 0
        except ZeroDivisionError:
            raise
        if not divisible:
            raise AssertionError("steps MUST be divisible by accumulate!")

    def _ensure_artifact_directory(self, artifact_base: str | None) -> None:
        if artifact_base is None:
            return
        directory = os.path.dirname(artifact_base)
        os.makedirs(directory, exist_ok=True)

    def _normalize_bucket_edges(self, bucket_edges):
        if bucket_edges is None:
            return None
        if torch.is_tensor(bucket_edges):
            if self.device is not None:
                return bucket_edges.to(self.device)
            return bucket_edges
        tensor = torch.as_tensor(bucket_edges, dtype=torch.float32)
        if self.device is not None:
            tensor = tensor.to(self.device)
        return tensor
