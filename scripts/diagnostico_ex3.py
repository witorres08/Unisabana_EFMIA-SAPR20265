"""
Ejercicio 3 — la evidencia, medida y no supuesta.

Este script produce los tres números que sustentan el diagnóstico documentado
en `DQNAgent.select_action`:

  1. Cuántos episodios de una política uniformemente aleatoria llegan a la
     bandera (respuesta: cero).
  2. Lo mismo, pero con exploración persistente (acciones repetidas).
  3. El "spread" de los Q-values de una red entrenada con exploración rota:
     ~0, es decir, la red concluye que ninguna acción importa.

    uv run python scripts/diagnostico_ex3.py
"""
import random

import gymnasium as gym
import numpy as np
import torch


def tasa_de_exito(n_episodios: int = 300, explore_repeat: int = 1, semilla: int = 0) -> tuple[int, float]:
    """Cuenta cuántos episodios TERMINAN (llegan a la bandera) en vez de agotar los 200 pasos.

    explore_repeat=1  -> epsilon-greedy de libro (acción nueva en cada paso)
    explore_repeat=k  -> la acción se sostiene entre 1 y k pasos
    """
    env = gym.make("MountainCar-v0")
    rng = random.Random(semilla)
    exitos, posiciones = 0, []

    for ep in range(n_episodios):
        env.reset(seed=semilla + ep)
        pendientes, accion = 0, 0
        max_pos, terminado, truncado = -1.2, False, False

        while not (terminado or truncado):
            if pendientes == 0:
                accion = rng.randrange(3)
                pendientes = rng.randint(1, explore_repeat)
            pendientes -= 1
            obs, _, terminado, truncado, _ = env.step(accion)
            max_pos = max(max_pos, float(obs[0]))

        exitos += int(terminado)
        posiciones.append(max_pos)

    env.close()
    return exitos, float(np.mean(posiciones))


def spread_de_q(agente) -> tuple[float, float]:
    """Q medio y separación media entre acciones dentro de un mismo estado."""
    env = gym.make("MountainCar-v0")
    estados = torch.as_tensor(
        np.array([env.observation_space.sample() for _ in range(200)]), dtype=torch.float32
    )
    env.close()
    with torch.no_grad():
        q = agente.q_net(estados.to(agente.device)).cpu().numpy()
    return float(q.mean()), float(np.abs(q.max(1) - q.min(1)).mean())


if __name__ == "__main__":
    print("=" * 74)
    print("1. ¿Cuántos episodios aleatorios llegan a la bandera?")
    print("=" * 74)
    print(f"{'explore_repeat':>16} | {'éxitos/300':>12} | {'posición máx. media':>20}")
    print("-" * 74)
    for k in (1, 5, 10, 20, 40):
        exitos, pos = tasa_de_exito(300, explore_repeat=k)
        etiqueta = f"{k}" + (" (de libro)" if k == 1 else "")
        print(f"{etiqueta:>16} | {exitos:>7}/300   | {pos:>20.3f}")
    print("\nLa bandera está en posición 0.5. Con k=1 el coche ni se acerca:")
    print("no es mala suerte, es que ese comportamiento le resulta inalcanzable.")
    print(f"\nP(misma acción 20 veces seguidas) = (1/3)^20 = {(1/3)**20:.2e}")

    print("\n" + "=" * 74)
    print("2. Q-values de una red entrenada con exploración rota (200 episodios)")
    print("=" * 74)
    from mountain_car.agents.dqn import DQNAgent

    roto = DQNAgent("MountainCar-v0", explore_repeat=1)   # exploración de libro
    roto.train(total_episodes=200, log_interval=100)
    media, spread = spread_de_q(roto)
    print(f"\nQ medio: {media:.2f}   |   separación entre acciones: {spread:.4f}")
    print(f"Punto fijo teórico si nunca se alcanza la meta: -1/(1-gamma) = {-1/(1-0.99):.0f}")
    print("\nLa media va hacia el punto fijo y la separación es ~0: la red aprendió")
    print("correctamente que da igual lo que haga. El fallo está aguas arriba.")
