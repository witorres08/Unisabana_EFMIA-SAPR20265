"""
Entrena ambos agentes, los evalúa sobre 100 episodios y guarda todo en
`results/` para que las gráficas y la documentación sean reproducibles.

    uv run python scripts/experimento.py
"""
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

from mountain_car.agents import DQNAgent, QLearningAgent

RESULTS = Path("results")
SAVES = Path("saves")
EPISODIOS_QL = 20_000
EPISODIOS_DQN = 2_500
EVAL_EPISODIOS = 100


def evaluar(agente, n_episodios: int = EVAL_EPISODIOS, semilla: int = 1234) -> dict:
    """Política puramente greedy sobre n episodios."""
    env = gym.make("MountainCar-v0")
    recompensas, pasos, exitos = [], [], 0
    for i in range(n_episodios):
        obs, _ = env.reset(seed=semilla + i)
        total, n, terminado, truncado = 0.0, 0, False, False
        while not (terminado or truncado):
            accion, _ = agente.predict(obs, deterministic=True)
            obs, r, terminado, truncado, _ = env.step(int(accion))
            total += r
            n += 1
        recompensas.append(total)
        pasos.append(n)
        exitos += int(terminado)
    env.close()
    return {
        "recompensa_media": float(np.mean(recompensas)),
        "recompensa_std": float(np.std(recompensas)),
        "recompensa_mejor": float(np.max(recompensas)),
        "recompensa_peor": float(np.min(recompensas)),
        "pasos_medios": float(np.mean(pasos)),
        "exitos": exitos,
        "n_episodios": n_episodios,
        "recompensas": [float(x) for x in recompensas],
    }


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    resumen = {}

    # ── Q-Learning tabular ────────────────────────────────────────────
    print("=" * 70)
    print(f"Q-LEARNING TABULAR — {EPISODIOS_QL} episodios")
    print("=" * 70)
    t0 = time.time()
    ql = QLearningAgent("MountainCar-v0")
    hist_ql = ql.train(total_episodes=EPISODIOS_QL, log_interval=2_000)
    t_ql = time.time() - t0
    ql.save(SAVES / "qlearning_mountaincar.pkl")

    eval_ql = evaluar(ql)
    resumen["qlearning"] = {
        **{k: v for k, v in eval_ql.items() if k != "recompensas"},
        "episodios_entrenamiento": EPISODIOS_QL,
        "tiempo_entrenamiento_s": round(t_ql, 1),
        "estados_visitados": len(ql.q_table),
        "estados_posibles": ql.n_bins ** 2,
        "parametros": len(ql.q_table) * ql.n_actions,
    }
    np.save(RESULTS / "hist_qlearning.npy", np.array(hist_ql))
    np.save(RESULTS / "eval_qlearning.npy", np.array(eval_ql["recompensas"]))
    print(f"\n-> {eval_ql['recompensa_media']:.2f} de media, "
          f"{eval_ql['exitos']}/{EVAL_EPISODIOS} llegan a la bandera ({t_ql:.0f}s)\n")

    # ── DQN ───────────────────────────────────────────────────────────
    print("=" * 70)
    print(f"DQN — {EPISODIOS_DQN} episodios")
    print("=" * 70)
    t0 = time.time()
    dqn = DQNAgent("MountainCar-v0")
    hist_dqn = dqn.train(total_episodes=EPISODIOS_DQN, log_interval=250)
    t_dqn = time.time() - t0
    dqn.save(SAVES / "dqn_mountaincar.pt")

    eval_dqn = evaluar(dqn)
    resumen["dqn"] = {
        **{k: v for k, v in eval_dqn.items() if k != "recompensas"},
        "episodios_entrenamiento": EPISODIOS_DQN,
        "tiempo_entrenamiento_s": round(t_dqn, 1),
        "parametros": sum(p.numel() for p in dqn.q_net.parameters()),
        "explore_repeat": dqn.explore_repeat,
    }
    np.save(RESULTS / "hist_dqn.npy", np.array(hist_dqn))
    np.save(RESULTS / "eval_dqn.npy", np.array(eval_dqn["recompensas"]))
    print(f"\n-> {eval_dqn['recompensa_media']:.2f} de media, "
          f"{eval_dqn['exitos']}/{EVAL_EPISODIOS} llegan a la bandera ({t_dqn:.0f}s)\n")

    with open(RESULTS / "resumen.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    print(f"Resultados guardados en {RESULTS}/")


if __name__ == "__main__":
    main()
