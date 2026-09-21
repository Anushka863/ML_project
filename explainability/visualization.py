"""
==============================================================================
ECG Attribution Visualization Module
==============================================================================
Generates publication-quality ECG waveform visualizations overlaid with
genuine model-derived Integrated Gradients attributions.
==============================================================================
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def plot_ecg_lead_attributions(
    lead_attributions: Dict[str, float],
    save_path: Optional[str] = None
) -> str:
    """
    Plots a horizontal bar chart of mean absolute Integrated Gradients attribution per ECG lead.
    """
    leads = list(lead_attributions.keys())
    values = [lead_attributions[k] for k in leads]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#dc2626" if v > np.median(values) else "#2563eb" for v in values]
    bars = ax.barh(leads, values, color=colors, edgecolor="none", height=0.6)

    ax.set_xlabel("Mean Absolute Integrated Gradients Attribution", fontsize=10, fontweight="bold")
    ax.set_title("PTB-XL Multimodal GNN: Model Attribution by ECG Lead", fontsize=12, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.invert_yaxis()

    for bar, val in zip(bars, values):
        ax.text(val + 1e-4, bar.get_y() + bar.get_height()/2, f"{val:.4f}",
                va="center", ha="left", fontsize=8, color="#334155")

    plt.tight_layout()
    output_path = save_path or "reports/figures/ecg_lead_attributions.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_ecg_waveform_with_attribution(
    ecg_waveform: np.ndarray,
    ecg_attribution: np.ndarray,
    selected_leads: Optional[List[int]] = None,
    save_path: Optional[str] = None
) -> str:
    """
    Plots selected ECG leads overlaid with model attribution intensity heatmap.
    ecg_waveform: shape (12, 1000)
    ecg_attribution: shape (12, 1000)
    """
    if ecg_waveform.ndim == 3:
        ecg_waveform = ecg_waveform[0]
    if ecg_attribution.ndim == 3:
        ecg_attribution = ecg_attribution[0]

    leads_to_plot = selected_leads or [0, 1, 6, 7]  # e.g. Leads I, II, V1, V2
    fig, axes = plt.subplots(len(leads_to_plot), 1, figsize=(10, 2.2 * len(leads_to_plot)), sharex=True)
    if len(leads_to_plot) == 1:
        axes = [axes]

    time_axis = np.linspace(0, 10, ecg_waveform.shape[1])  # 10-second window

    for ax, lead_idx in zip(axes, leads_to_plot):
        lead_name = LEAD_NAMES[lead_idx] if lead_idx < len(LEAD_NAMES) else f"Lead {lead_idx}"
        wave = ecg_waveform[lead_idx]
        attr = np.abs(ecg_attribution[lead_idx])

        # Normalize attribution for alpha shading
        max_attr = np.max(attr) + 1e-8
        norm_attr = attr / max_attr

        ax.plot(time_axis, wave, color="#0f172a", lw=1.2, label=f"{lead_name} Signal")
        
        # Color-coded attribution shading
        ax.fill_between(time_axis, wave, np.min(wave), where=norm_attr > 0.4,
                        color="#ef4444", alpha=0.35, label="High Attribution Region (>40% max)")
        ax.set_ylabel(lead_name, fontweight="bold", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        if ax == axes[0]:
            ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (seconds, 100Hz)", fontsize=10, fontweight="bold")
    fig.suptitle("Genuine Model-Derived ECG Attribution (Integrated Gradients)", fontsize=12, fontweight="bold", y=0.99)
    plt.tight_layout()

    output_path = save_path or "reports/figures/ecg_attribution_overlay.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path
