# Copyright (c) 2025, Your Name
# All rights reserved.

from __future__ import annotations

import os
import pathlib
import statistics
import time
import torch
from collections import deque
from typing import Any


class Logger:
    """Logger to save the learning metrics to console and/or TensorBoard.

    By default, only console logging is used. To enable TensorBoard logging,
    pass ``log_dir`` and set ``logger_type="tensorboard"``.
    """

    def __init__(
        self,
        log_dir: str | None = None,
        cfg: dict | None = None,
        num_envs: int = 1,
        device: str = "cpu",
        logger_type: str = "console",
    ) -> None:
        """Initialize buffers and logging state for a training run.

        Args:
            log_dir: Directory for logging output. If None, no file writer is used.
            cfg: Configuration dictionary.
            num_envs: Number of parallel environments.
            device: Device for tensor operations.
            logger_type: Type of logger: ``"console"`` (default), ``"tensorboard"``.
        """
        self.log_dir = log_dir
        self.cfg = cfg or {}
        self.num_envs = num_envs
        self.device = device
        self.tot_timesteps = 0
        self.tot_time = 0

        # Create buffers
        self.ep_extras: list[dict] = []
        self.rewbuffer: deque[float] = deque(maxlen=100)
        self.lenbuffer: deque[float] = deque(maxlen=100)
        self.per_step_rewbuffer: deque[float] = deque(maxlen=100)
        self.cur_reward_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.cur_episode_length = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # Initialize writer
        self.logger_type = logger_type.lower()
        self.writer = None
        self._init_logging_writer()

    def _init_logging_writer(self) -> None:
        """Initialize the logging writer."""
        if self.log_dir is not None and self.logger_type == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

    def process_env_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict,
    ) -> None:
        """Add metrics from the environment step to the buffers.

        Args:
            rewards: Rewards tensor of shape ``(num_envs,)``.
            dones: Done flags of shape ``(num_envs,)``.
            extras: Extra information dict from the environment.
        """
        if "episode" in extras:
            self.ep_extras.append(extras["episode"])
        elif "log" in extras:
            self.ep_extras.append(extras["log"])

        # Track the per-step reward so the user can see the actual reward
        # magnitude per environment step. The "Mean reward" log line is
        # the *episode total* (sum of per-step rewards across the entire
        # episode), which can be misleading when the env returns
        # large-magnitude penalties.
        self.per_step_rewbuffer.extend(rewards.detach().cpu().numpy().tolist())

        self.cur_reward_sum += rewards
        self.cur_episode_length += 1

        new_ids = (dones > 0).nonzero(as_tuple=False)
        if new_ids.numel() > 0:
            self.rewbuffer.extend(self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
            self.lenbuffer.extend(self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
            self.cur_reward_sum[new_ids] = 0
            self.cur_episode_length[new_ids] = 0

    def log(
        self,
        it: int,
        start_it: int,
        total_it: int,
        collect_time: float,
        learn_time: float,
        loss_dict: dict,
        learning_rate: float,
        action_std: torch.Tensor,
        num_steps_per_env: int = 1,
        print_minimal: bool = False,
        width: int = 80,
        pad: int = 40,
    ) -> None:
        """Log the training metrics to the logging service and print to console.

        Args:
            it: Current iteration number.
            start_it: Starting iteration number.
            total_it: Total number of iterations.
            collect_time: Time spent collecting rollouts.
            learn_time: Time spent learning.
            loss_dict: Dictionary of loss values.
            learning_rate: Current learning rate.
            action_std: Current action standard deviation.
            num_steps_per_env: Number of steps collected per environment.
            print_minimal: Whether to print minimal output.
            width: Width of the log box.
            pad: Padding for log alignment.
        """
        collection_size = num_steps_per_env * self.num_envs
        iteration_time = collect_time + learn_time
        self.tot_timesteps += collection_size
        self.tot_time += iteration_time

        # Log to writer if available
        if self.writer is not None:
            extras_string = self._log_to_writer(it, collect_time, learn_time, loss_dict, learning_rate, action_std)

        # Print to console
        self._print_to_console(
            it,
            start_it,
            total_it,
            collect_time,
            learn_time,
            loss_dict,
            learning_rate,
            action_std,
            collection_size,
            iteration_time,
            print_minimal,
            width,
            pad,
        )

        # Clear extras buffer
        self.ep_extras.clear()

    def _log_to_writer(
        self,
        it: int,
        collect_time: float,
        learn_time: float,
        loss_dict: dict,
        learning_rate: float,
        action_std: torch.Tensor,
    ) -> str:
        """Log metrics to the TensorBoard writer."""
        collection_size = self.cfg.get("num_steps_per_env", 1) * self.num_envs
        extras_string = ""

        # Log episode extras
        if self.ep_extras:
            for key in {k for ep_info in self.ep_extras for k in ep_info}:
                infotensor = torch.empty(0, device=self.device)
                for ep_info in self.ep_extras:
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.as_tensor([ep_info[key]], device=self.device)
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    if ep_info[key].numel() > 0:
                        infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                if infotensor.numel() > 0:
                    value = torch.mean(infotensor)
                    if "/" in key:
                        self.writer.add_scalar(key, value, it)
                        extras_string += f"""{f"{key}:":>40} {value:.4f}\n"""
                    else:
                        self.writer.add_scalar("Episode/" + key, value, it)
                        extras_string += f"""{f"Mean episode {key}:":>40} {value:.4f}\n"""

        # Log losses
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.writer.add_scalar(f"Loss/{key}", value, it)
        self.writer.add_scalar("Loss/learning_rate", learning_rate, it)

        # Log std
        if isinstance(action_std, torch.Tensor):
            self.writer.add_scalar("Policy/mean_std", action_std.mean().item(), it)

        # Log performance
        fps = int(collection_size / (collect_time + learn_time)) if (collect_time + learn_time) > 0 else 0
        self.writer.add_scalar("Perf/total_fps", fps, it)
        self.writer.add_scalar("Perf/collection_time", collect_time, it)
        self.writer.add_scalar("Perf/learning_time", learn_time, it)

        # Log rewards and episode length
        if len(self.rewbuffer) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(self.rewbuffer), it)
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(self.lenbuffer), it)
            self.writer.add_scalar("Train/mean_reward/time", statistics.mean(self.rewbuffer), int(self.tot_time))
            self.writer.add_scalar(
                "Train/mean_episode_length/time", statistics.mean(self.lenbuffer), int(self.tot_time)
            )
        if len(self.per_step_rewbuffer) > 0:
            self.writer.add_scalar("Train/mean_per_step_reward", statistics.mean(self.per_step_rewbuffer), it)

        return extras_string

    def _print_to_console(
        self,
        it: int,
        start_it: int,
        total_it: int,
        collect_time: float,
        learn_time: float,
        loss_dict: dict,
        learning_rate: float,
        action_std: torch.Tensor,
        collection_size: int,
        iteration_time: float,
        print_minimal: bool,
        width: int,
        pad: int,
    ) -> None:
        """Print training metrics to console."""
        fps = int(collection_size / iteration_time) if iteration_time > 0 else 0

        log_string = f"""{'#' * width}\n"""
        log_string += f"""\033[1m{f" Learning iteration {it}/{total_it} ".center(width)}\033[0m \n\n"""

        # Print run name if provided
        run_name = self.cfg.get("run_name")
        if run_name:
            log_string += f"""{"Run name:":>{pad}} {run_name}\n"""

        # Print performance
        log_string += (
            f"""{"Total steps:":>{pad}} {self.tot_timesteps} \n"""
            f"""{"Steps per second:":>{pad}} {fps:.0f} \n"""
            f"""{"Collection time:":>{pad}} {collect_time:.3f}s \n"""
            f"""{"Learning time:":>{pad}} {learn_time:.3f}s \n"""
        )

        # Print losses
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            log_string += f"""{f"Mean {key} loss:":>{pad}} {value:.4f}\n"""

        # Log rewards and episode length
        if len(self.rewbuffer) > 0:
            log_string += f"""{"Mean reward:":>{pad}} {statistics.mean(self.rewbuffer):.2f}\n"""
            log_string += f"""{"Mean episode length:":>{pad}} {statistics.mean(self.lenbuffer):.2f}\n"""
        if len(self.per_step_rewbuffer) > 0:
            # Per-step reward is the actual value the agent receives each
            # env.step. Should be small magnitude (e.g. -1 to +1) for a
            # well-tuned env; large values (e.g. -100) indicate either
            # bad reward shaping or the need for reward normalization.
            log_string += f"""{"Mean per-step reward:":>{pad}} {statistics.mean(self.per_step_rewbuffer):.4f}\n"""

        # Print std
        if isinstance(action_std, torch.Tensor):
            log_string += f"""{"Mean action std:":>{pad}} {action_std.mean().item():.2f}\n"""

        # Print footer
        done_it = it + 1 - start_it
        remaining_it = total_it - start_it - done_it
        if done_it > 0:
            eta = self.tot_time / done_it * remaining_it
        else:
            eta = 0
        log_string += (
            f"""{"-" * width}\n"""
            f"""{"Iteration time:":>{pad}} {iteration_time:.2f}s\n"""
            f"""{"Time elapsed:":>{pad}} {time.strftime('%H:%M:%S', time.gmtime(self.tot_time))}\n"""
            f"""{"ETA:":>{pad}} {time.strftime('%H:%M:%S', time.gmtime(eta))}\n"""
        )
        print(log_string)

    def save_model(self, path: str, it: int) -> None:
        """Save model checkpoint. (Hook for external logging integration.)"""
        pass

    def stop_logging_writer(self) -> None:
        """Stop the logging writer."""
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def close(self) -> None:
        """Alias for stop_logging_writer."""
        self.stop_logging_writer()


class TensorboardLogger(Logger):
    """Convenience subclass that always uses TensorBoard logging.

    Usage::

        logger = TensorboardLogger(log_dir="/path/to/logs")
    """

    def __init__(
        self,
        log_dir: str,
        cfg: dict | None = None,
        num_envs: int = 1,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            log_dir=log_dir,
            cfg=cfg,
            num_envs=num_envs,
            device=device,
            logger_type="tensorboard",
        )