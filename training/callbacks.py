"""Training callbacks: WandB logging integrated with EvalCallback.

Provides `WandbEvalCallback` which wraps SB3 EvalCallback metrics into Weights & Biases.
"""

from __future__ import annotations

import os
import json
import shutil
from datetime import datetime
from typing import Any, Dict, Optional

from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import VecNormalize


class WandbEvalCallback(EvalCallback):
    """Eval callback that logs key metrics to wandb if enabled and
    saves VecNormalize stats when a new best model is found.

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

    def _on_step(self) -> bool:
        prev_best = float(getattr(self, "best_mean_reward", float("-inf")))
        continue_training = super()._on_step()

        # Check if evaluation just occurred
        eval_freq = int(getattr(self, "eval_freq", 0))
        n_calls = int(getattr(self, "n_calls", 0))
        eval_happened = eval_freq > 0 and n_calls % eval_freq == 0

        if eval_happened:
            curr_best = float(getattr(self, "best_mean_reward", float("-inf")))
            
            # Save VecNormalize if new best found
            if (self.best_model_save_path and curr_best > prev_best 
                and isinstance(self._vecnorm_env, VecNormalize)):
                os.makedirs(self.best_model_save_path, exist_ok=True)
                save_path = os.path.join(self.best_model_save_path, "vecnorm_best.pkl")
                print(f"[CALLBACK] New best: {prev_best:.3f} -> {curr_best:.3f}; saving VecNormalize")
                self._vecnorm_env.save(save_path)
            elif curr_best > prev_best:
                print(f"[CALLBACK] New best: {prev_best:.3f} -> {curr_best:.3f}")

            # Log to WandB
            if self._wandb is not None:
                logs = {
                    "eval/mean_reward": float(getattr(self, "last_mean_reward", 0)),
                    "time/total_timesteps": float(self.num_timesteps),
                }
                if len(self._is_success_buffer) > 0:
                    logs["eval/success_rate"] = sum(self._is_success_buffer) / len(self._is_success_buffer)
                
                import wandb
                self._wandb.log(logs)

        return continue_training


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
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_saved_at < self.save_freq_steps:
            return True
        self._save_checkpoint()
        self._last_saved_at = self.num_timesteps
        return True

    def _save_checkpoint(self) -> None:
        step_dir = os.path.join(self.save_dir, f"{self.prefix}_step_{self.num_timesteps}")
        os.makedirs(step_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(step_dir, "model")
        self.model.save(model_path)
        
        # Save vecnorm if available
        vecnorm_path = None
        if isinstance(self.vecnorm_env, VecNormalize):
            vecnorm_path = os.path.join(step_dir, "vecnorm.pkl")
            self.vecnorm_env.save(vecnorm_path)
        
        # Save metadata
        self._save_metadata(step_dir, model_path, vecnorm_path)
        
        # Upload to WandB if requested
        if self.to_wandb and self.wandb_run is not None:
            self._upload_to_wandb(model_path, vecnorm_path, step_dir)
        
        # Clean old checkpoints
        self._prune_old_checkpoints()

    def _save_metadata(self, step_dir: str, model_path: str, vecnorm_path: Optional[str]) -> None:
        meta_path = os.path.join(step_dir, "meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "step": int(self.num_timesteps),
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "model": os.path.basename(model_path) + ".zip",
                "vecnorm": os.path.basename(vecnorm_path) if vecnorm_path else None,
            }, f, indent=2)

    def _upload_to_wandb(self, model_path: str, vecnorm_path: Optional[str], step_dir: str) -> None:
        import wandb
        
        art = wandb.Artifact(f"checkpoint_step_{self.num_timesteps}", type="model")
        art.add_file(model_path + ".zip")
        if vecnorm_path and os.path.exists(vecnorm_path):
            art.add_file(vecnorm_path)
        art.add_file(os.path.join(step_dir, "meta.json"))
        self.wandb_run.log_artifact(art)

    def _prune_old_checkpoints(self) -> None:
        if self.keep_last_k <= 0:
            return
        
        import re
        pattern = re.compile(rf"^{re.escape(self.prefix)}_step_(\d+)$")
        
        entries = []
        for name in os.listdir(self.save_dir):
            full = os.path.join(self.save_dir, name)
            if not os.path.isdir(full):
                continue
            m = pattern.match(name)
            if m:
                step = int(m.group(1))
                entries.append((step, full))
        
        entries.sort(key=lambda x: x[0], reverse=True)
        for _, path in entries[self.keep_last_k:]:
            shutil.rmtree(path, ignore_errors=True)


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
            if isinstance(info, dict) and "reward_terms" in info:
                rt = info["reward_terms"]
                if isinstance(rt, dict):
                    self._log_reward_terms(rt)
        return True

    def _log_reward_terms(self, reward_terms: Dict[str, Any]) -> None:
        # Log to SB3 logger
        total = float(reward_terms.get("total", 0))
        self.logger.record(f"{self._prefix}/total", total)
        
        for category in ["contrib", "raw"]:
            terms = reward_terms.get(category, {}) or {}
            for k, v in terms.items():
                self.logger.record(f"{self._prefix}/{category}/{k}", float(v))
        
        # Log to WandB
        if self._wandb is not None:
            import wandb
            
            data = {f"{self._prefix}/total": total}
            for category in ["contrib", "raw"]:
                terms = reward_terms.get(category, {}) or {}
                for k, v in terms.items():
                    data[f"{self._prefix}/{category}/{k}"] = float(v)
            self._wandb.log(data)

