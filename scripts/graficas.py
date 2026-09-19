"""
Genera las figuras del Paso 4 a partir de lo que dejó `experimento.py`.

    uv run python scripts/graficas.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

RESULTS = Path("results")
C_QL, C_DQN = "#C1440E", "#1F5C8B"


def media_movil(x: np.ndarray, ventana: int) -> np.ndarray:
    if len(x) < ventana:
        return x
    return np.convolve(x, np.ones(ventana) / ventana, mode="valid")


def fig_curvas(hist_ql: np.ndarray, hist_dqn: np.ndarray) -> None:
    """Curvas de aprendizaje. Ejes separados: las escalas de episodios difieren 8x."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

    ax1.plot(hist_ql, color=C_QL, alpha=.15, lw=.5)
    v = 200
    ax1.plot(np.arange(v - 1, len(hist_ql)), media_movil(hist_ql, v), color=C_QL, lw=2,
             label=f"media móvil ({v} ep.)")
    ax1.set_title("Q-Learning tabular")
    ax1.set_xlabel("Episodio")
    ax1.set_ylabel("Recompensa del episodio")

    ax2.plot(hist_dqn, color=C_DQN, alpha=.15, lw=.5)
    v = 50
    ax2.plot(np.arange(v - 1, len(hist_dqn)), media_movil(hist_dqn, v), color=C_DQN, lw=2,
             label=f"media móvil ({v} ep.)")
    ax2.set_title("DQN")
    ax2.set_xlabel("Episodio")

    for ax in (ax1, ax2):
        ax.axhline(-110, ls="--", c="green", lw=1.2, label="umbral 'resuelto' (-110)")
        ax.axhline(-200, ls=":", c="gray", lw=1.2, label="suelo (-200, nunca llega)")
        ax.set_ylim(-210, -80)
        ax.grid(alpha=.3)
        ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("Curvas de aprendizaje — MountainCar-v0", fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(RESULTS / "curvas_aprendizaje.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_curvas_normalizadas(hist_ql: np.ndarray, hist_dqn: np.ndarray) -> None:
    """Las dos curvas en un mismo eje, en % del entrenamiento, para comparar la forma."""
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for hist, color, nombre, v in ((hist_ql, C_QL, "Q-Learning tabular", 200),
                                   (hist_dqn, C_DQN, "DQN", 50)):
        suave = media_movil(hist, v)
        x = np.linspace(0, 100, len(suave))
        ax.plot(x, suave, color=color, lw=2, label=f"{nombre} ({len(hist):,} ep.)")
    ax.axhline(-110, ls="--", c="green", lw=1.2, label="umbral 'resuelto'")
    ax.set_xlabel("% del entrenamiento completado")
    ax.set_ylabel("Recompensa (media móvil)")
    ax.set_title("Misma comparación en escala relativa")
    ax.set_ylim(-210, -80)
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "curvas_normalizadas.png", dpi=140)
    plt.close(fig)


def fig_evaluacion(ev_ql: np.ndarray, ev_dqn: np.ndarray) -> None:
    """Distribución de la recompensa en los 100 episodios de evaluación greedy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2),
                                   gridspec_kw={"width_ratios": [2, 1]})
    bins = np.arange(-200, -80, 5)
    ax1.hist(ev_ql, bins=bins, alpha=.7, color=C_QL, label=f"Q-Learning (μ={ev_ql.mean():.1f})")
    ax1.hist(ev_dqn, bins=bins, alpha=.7, color=C_DQN, label=f"DQN (μ={ev_dqn.mean():.1f})")
    ax1.axvline(-110, ls="--", c="green", lw=1.5, label="umbral 'resuelto'")
    ax1.set_xlabel("Recompensa del episodio")
    ax1.set_ylabel("Frecuencia")
    ax1.set_title("Distribución en 100 episodios de evaluación greedy")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=.3)

    bp = ax2.boxplot([ev_ql, ev_dqn], tick_labels=["Q-Learning", "DQN"], patch_artist=True, widths=.5)
    for parche, color in zip(bp["boxes"], (C_QL, C_DQN)):
        parche.set_facecolor(color)
        parche.set_alpha(.7)
    ax2.axhline(-110, ls="--", c="green", lw=1.2)
    ax2.set_ylabel("Recompensa")
    ax2.set_title("Dispersión")
    ax2.grid(alpha=.3, axis="y")

    fig.tight_layout()
    fig.savefig(RESULTS / "evaluacion.png", dpi=140)
    plt.close(fig)


def fig_politicas() -> None:
    """Política greedy de cada agente sobre el plano (posición, velocidad)."""
    from mountain_car.agents import DQNAgent, QLearningAgent

    ql = QLearningAgent.load(Path("saves/qlearning_mountaincar.pkl"))
    dqn = DQNAgent.load(Path("saves/dqn_mountaincar.pt"))

    n = 200
    pos = np.linspace(-1.2, 0.6, n)
    vel = np.linspace(-0.07, 0.07, n)
    P, V = np.meshgrid(pos, vel)
    obs = np.stack([P.ravel(), V.ravel()], axis=1).astype(np.float32)

    # Q-Learning: una consulta por celda de la rejilla
    acc_ql = np.array([ql.select_action(ql.discretize(o), deterministic=True) for o in obs])
    # DQN: un solo forward para toda la malla
    with torch.no_grad():
        t = torch.as_tensor(obs, dtype=torch.float32, device=dqn.device)
        acc_dqn = dqn.q_net(t).argmax(dim=1).cpu().numpy()

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#C1440E", "#EAEAEA", "#1F5C8B"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, acc, titulo in ((axes[0], acc_ql, "Q-Learning tabular (rejilla 20×20)"),
                            (axes[1], acc_dqn, "DQN (red continua)")):
        im = ax.pcolormesh(P, V, acc.reshape(n, n), cmap=cmap, vmin=0, vmax=2, shading="auto")
        ax.set_xlabel("Posición")
        ax.set_title(titulo)
        ax.axvline(0.5, c="green", ls="--", lw=1.2)
    axes[0].set_ylabel("Velocidad")
    cbar = fig.colorbar(im, ax=axes, ticks=[0.33, 1, 1.67], fraction=.03)
    cbar.ax.set_yticklabels(["izquierda", "nada", "derecha"])
    fig.suptitle("Política greedy aprendida — el escalonado del tabular es la discretización",
                 fontsize=12)
    fig.savefig(RESULTS / "politicas.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    hist_ql = np.load(RESULTS / "hist_qlearning.npy")
    hist_dqn = np.load(RESULTS / "hist_dqn.npy")
    ev_ql = np.load(RESULTS / "eval_qlearning.npy")
    ev_dqn = np.load(RESULTS / "eval_dqn.npy")

    fig_curvas(hist_ql, hist_dqn)
    fig_curvas_normalizadas(hist_ql, hist_dqn)
    fig_evaluacion(ev_ql, ev_dqn)
    fig_politicas()

    with open(RESULTS / "resumen.json", encoding="utf-8") as f:
        resumen = json.load(f)
    print(json.dumps(resumen, indent=2, ensure_ascii=False))
    print(f"\nFiguras escritas en {RESULTS}/")


if __name__ == "__main__":
    main()
