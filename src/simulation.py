from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .dataset import build_window_spec


def generate_synthetic_pendulum_dataframe(
    simulation_config: dict[str, Any],
    data_config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    window_spec = build_window_spec(data_config)
    fps = int(data_config["fps"])
    num_samples = int(simulation_config["num_samples"])
    total_duration_seconds = float(simulation_config["duration_seconds"])
    total_steps = int(round(fps * total_duration_seconds))

    if total_steps < window_spec.total_steps:
        raise ValueError(
            "Simulation duration is shorter than the required input + prediction window: "
            f"total_steps={total_steps}, required={window_spec.total_steps}."
        )

    physical = simulation_config["physical"]
    initial_state = simulation_config["initial_state"]

    gravity = float(physical["gravity"])
    length = float(physical["length"])
    mass = float(physical["mass"])
    linear_damping_base = float(physical["linear_damping_base"])
    quadratic_damping_base = float(physical["quadratic_damping_base"])
    damping_uncertainty_ratio = float(simulation_config["damping_uncertainty_ratio"])
    noise_std = float(simulation_config["noise_std"])

    linear_min = linear_damping_base * (1.0 - damping_uncertainty_ratio)
    linear_max = linear_damping_base * (1.0 + damping_uncertainty_ratio)
    quadratic_min = quadratic_damping_base * (1.0 - damping_uncertainty_ratio)
    quadratic_max = quadratic_damping_base * (1.0 + damping_uncertainty_ratio)

    linear_damping_values = np.random.uniform(linear_min, linear_max, size=num_samples)
    quadratic_damping_values = np.random.uniform(quadratic_min, quadratic_max, size=num_samples)

    theta = np.full(num_samples, float(initial_state["theta"]), dtype=np.float64)
    omega = np.full(num_samples, float(initial_state["omega"]), dtype=np.float64)
    dt = 1.0 / fps

    clean_sequences = np.zeros((num_samples, total_steps), dtype=np.float32)

    for step in range(total_steps):
        clean_sequences[:, step] = theta.astype(np.float32)

        k1_theta, k1_omega = _pendulum_derivatives(
            theta, omega, linear_damping_values, quadratic_damping_values, gravity, length, mass
        )
        k2_theta, k2_omega = _pendulum_derivatives(
            theta + 0.5 * dt * k1_theta,
            omega + 0.5 * dt * k1_omega,
            linear_damping_values,
            quadratic_damping_values,
            gravity,
            length,
            mass,
        )
        k3_theta, k3_omega = _pendulum_derivatives(
            theta + 0.5 * dt * k2_theta,
            omega + 0.5 * dt * k2_omega,
            linear_damping_values,
            quadratic_damping_values,
            gravity,
            length,
            mass,
        )
        k4_theta, k4_omega = _pendulum_derivatives(
            theta + dt * k3_theta,
            omega + dt * k3_omega,
            linear_damping_values,
            quadratic_damping_values,
            gravity,
            length,
            mass,
        )

        theta += (dt / 6.0) * (k1_theta + 2.0 * k2_theta + 2.0 * k3_theta + k4_theta)
        omega += (dt / 6.0) * (k1_omega + 2.0 * k2_omega + 2.0 * k3_omega + k4_omega)

    noise = np.random.normal(loc=0.0, scale=noise_std, size=clean_sequences.shape).astype(np.float32)
    noisy_sequences = clean_sequences + noise

    column_names = [f"t_{step:05d}" for step in range(total_steps)]
    frame = pd.DataFrame(noisy_sequences, columns=column_names)
    if bool(simulation_config.get("write_sample_id_column", True)):
        frame.insert(0, "sample_id", [f"sim_{index:05d}" for index in range(num_samples)])

    metadata = {
        "num_samples": num_samples,
        "fps": fps,
        "total_duration_seconds": total_duration_seconds,
        "total_steps": total_steps,
        "input_seconds": window_spec.input_seconds,
        "pred_seconds": window_spec.pred_seconds,
        "input_steps": window_spec.input_steps,
        "pred_steps": window_spec.pred_steps,
        "noise_std": noise_std,
        "damping_uncertainty_ratio": damping_uncertainty_ratio,
        "linear_damping_range": [linear_min, linear_max],
        "quadratic_damping_range": [quadratic_min, quadratic_max],
    }
    return frame, metadata


def _pendulum_derivatives(
    theta: np.ndarray,
    omega: np.ndarray,
    linear_damping: np.ndarray,
    quadratic_damping: np.ndarray,
    gravity: float,
    length: float,
    mass: float,
) -> tuple[np.ndarray, np.ndarray]:
    theta_derivative = omega
    omega_derivative = (
        -(gravity / length) * np.sin(theta)
        - (linear_damping / mass) * omega
        - (quadratic_damping * length / mass) * omega * np.abs(omega)
    )
    return theta_derivative, omega_derivative
