"""Training callbacks: WandB logging integrated with EvalCallback.

Provides `WandbEvalCallback` which wraps SB3 EvalCallback metrics into Weights & Biases.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from stable_baselines3.common.callbacks import EvalCallback, BaseCallback


class WandbEvalCallback(EvalCallback):
    """Eval callback that logs key metrics to wandb if enabled.

    Usage: pass an initialized `wandb.Run` via `wandb_run`.
    """

    def __init__(
        self,
        *args,
        wandb_run: Optional[Any] = None,
        vecnorm_env: Optional[Any] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._wandb = wandb_run
        # Optional reference to the training VecNormalize instance (or wrapped env)
        self._vecnorm_env = vecnorm_env
        # No need to track prev best; we'll always sync vecnorm alongside best

    def _on_step(self) -> bool:
        return super()._on_step()

    def _on_event(self) -> None:
        # Capture previous best before EvalCallback updates
        prev_best = float(getattr(self, "best_mean_reward", float("-inf")))
        super()._on_event()
        # Save VecNormalize only when best improved in this event
        try:
            from stable_baselines3.common.vec_env import VecNormalize as _VN  # type: ignore

            curr_best = float(getattr(self, "best_mean_reward", float("-inf")))
            if (
                self.best_model_save_path
                and curr_best > prev_best
                and isinstance(self._vecnorm_env, _VN)
            ):
                import os

                os.makedirs(self.best_model_save_path, exist_ok=True)
                save_path = os.path.join(self.best_model_save_path, "vecnorm_best.pkl")
                print(f"[CALLBACK] New best reward: {prev_best:.3f} -> {curr_best:.3f}, saving VecNormalize to {save_path}")
                try:
                    self._vecnorm_env.save(save_path)
                    print(f"[CALLBACK] Successfully saved VecNormalize to {save_path}")
                except Exception as e:
                    print(f"[CALLBACK] Failed to save VecNormalize to {save_path}: {e}")
            elif curr_best > prev_best:
                print(f"[CALLBACK] New best reward: {prev_best:.3f} -> {curr_best:.3f}, but no VecNormalize to save")
        except Exception as e:
            print(f"[CALLBACK] Error in VecNormalize save logic: {e}")
        if self._wandb is None:
            return
        # After each evaluation is complete, EvalCallback sets these attributes
        # - self.last_mean_reward
        # - self.best_mean_reward
        # try to log success rate if present in info dicts
        logs: Dict[str, float] = {
            "eval/mean_reward": float(getattr(self, "last_mean_reward", float("nan"))),
            "eval/best_mean_reward": float(
                getattr(self, "best_mean_reward", float("nan"))
            ),
            "time/total_timesteps": float(self.num_timesteps),
        }
        try:
            # success rates collected by EvalCallback via info["is_success"]
            if len(self._is_success_buffer) > 0:
                logs["eval/success_rate"] = float(
                    sum(self._is_success_buffer) / len(self._is_success_buffer)
                )
        except Exception:
            pass
        try:
            import wandb  # type: ignore

            self._wandb.log(logs)
        except Exception:
            pass


class CheckpointCallbackWithVecnorm(BaseCallback):
    """Periodic checkpoint saver for model and VecNormalize stats.

    Saves every `save_freq_steps` timesteps to directories under `save_dir` with the pattern:
      {save_dir}/{prefix}_step_{num_timesteps}/ { model.zip, vecnorm.pkl, meta.json }
    Optionally uploads a W&B artifact per checkpoint.
    """

    def __init__(
        self,
        save_freq_steps: int,
        save_dir: str,
        vecnorm_env: Optional[Any] = None,
        keep_last_k: int = 0,
        prefix: str = "ckpt",
        to_wandb: bool = False,
        wandb_run: Optional[Any] = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.save_freq_steps = int(save_freq_steps)
        self.save_dir = save_dir
        self.vecnorm_env = vecnorm_env
        self.keep_last_k = int(keep_last_k)
        self.prefix = prefix
        self.to_wandb = to_wandb
        self.wandb_run = wandb_run
        self._last_saved_at: int = 0

    def _init_callback(self) -> None:
        import os

        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_saved_at < self.save_freq_steps:
            return True
        self._save_checkpoint()
        self._last_saved_at = self.num_timesteps
        return True

    def _save_checkpoint(self) -> None:
        import os
        import json
        from datetime import datetime
        from stable_baselines3.common.vec_env import VecNormalize as _VN  # type: ignore

        step_dir = os.path.join(
            self.save_dir, f"{self.prefix}_step_{self.num_timesteps}"
        )
        try:
            os.makedirs(step_dir, exist_ok=True)
            # Save model
            model_path = os.path.join(step_dir, "model")
            self.model.save(model_path)
            # Save vecnorm if available
            vecnorm_path = None
            if isinstance(self.vecnorm_env, _VN):
                vecnorm_path = os.path.join(step_dir, "vecnorm.pkl")
                try:
                    self.vecnorm_env.save(vecnorm_path)
                except Exception:
                    vecnorm_path = None
            # Save meta
            meta_path = os.path.join(step_dir, "meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "step": int(self.num_timesteps),
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "model": os.path.basename(model_path) + ".zip",
                        "vecnorm": (
                            os.path.basename(vecnorm_path) if vecnorm_path else None
                        ),
                    },
                    f,
                    indent=2,
                )
            # Optional: upload to WandB
            if self.to_wandb and self.wandb_run is not None:
                try:
                    import wandb  # type: ignore

                    art = wandb.Artifact(
                        f"checkpoint_step_{self.num_timesteps}", type="model"
                    )
                    art.add_file(model_path + ".zip")
                    if vecnorm_path is not None and os.path.exists(vecnorm_path):
                        art.add_file(vecnorm_path)
                    art.add_file(meta_path)
                    self.wandb_run.log_artifact(art)
                except Exception:
                    pass
        except Exception:
            pass


class RewardTermsLoggingCallback(BaseCallback):
    """Logs reward breakdown (total, contrib, raw) when provided by env infos.

    Works with VecEnv: inspects `self.locals["infos"]` each step and, when a
    dict contains `reward_terms`, logs it to TensorBoard and optionally WandB.
    """

    def __init__(
        self,
        wandb_run: Optional[Any] = None,
        prefix: str = "train/reward",
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self._wandb = wandb_run
        self._prefix = prefix

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not isinstance(infos, (list, tuple)):
            return True
        for info in infos:
            if not isinstance(info, dict):
                continue
            rt = info.get("reward_terms")
            if not isinstance(rt, dict):
                continue
            # Log to SB3 logger (TB/CSV if configured)
            try:
                total = float(rt.get("total", float("nan")))
                self.logger.record(f"{self._prefix}/total", total)
                contrib = rt.get("contrib", {}) or {}
                raw = rt.get("raw", {}) or {}
                for k, v in contrib.items():
                    self.logger.record(f"{self._prefix}/contrib/{k}", float(v))
                for k, v in raw.items():
                    self.logger.record(f"{self._prefix}/raw/{k}", float(v))
            except Exception:
                pass
            # Log to WandB if enabled
            if self._wandb is not None:
                try:
                    import wandb  # type: ignore

                    data = {
                        f"{self._prefix}/total": float(rt.get("total", float("nan")))
                    }
                    for k, v in (rt.get("contrib", {}) or {}).items():
                        data[f"{self._prefix}/contrib/{k}"] = float(v)
                    for k, v in (rt.get("raw", {}) or {}).items():
                        data[f"{self._prefix}/raw/{k}"] = float(v)
                    self._wandb.log(data)
                except Exception:
                    pass
        return True

    def _prune_old_checkpoints(self) -> None:
        if self.keep_last_k <= 0:
            return
        import os
        import re

        pattern = re.compile(rf"^{re.escape(self.prefix)}_step_(\d+)$")
        try:
            entries = []
            for name in os.listdir(self.save_dir):
                full = os.path.join(self.save_dir, name)
                if not os.path.isdir(full):
                    continue
                m = pattern.match(name)
                if not m:
                    continue
                step = int(m.group(1))
                entries.append((step, full))
            entries.sort(key=lambda x: x[0], reverse=True)
            for _, path in entries[self.keep_last_k :]:
                try:
                    # Best-effort remove directory tree
                    import shutil

                    shutil.rmtree(path, ignore_errors=True)
                except Exception:
                    pass
        except Exception:
            pass
