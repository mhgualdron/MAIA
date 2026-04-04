# RL — Notebooks: Laboratorios y Tutoriales

Colección de notebooks de práctica del módulo de Reinforcement Learning.

## Contenido

| Notebook | Tópico |
|----------|--------|
| [`tutorial_bandidos.ipynb`](./tutorial_bandidos.ipynb) | Introducción al problema del bandido multi-brazo |
| [`bandidos_epsilon.ipynb`](./bandidos_epsilon.ipynb) | Algoritmo ε-greedy para bandidos |
| [`gridworld_mdps_tutorial.ipynb`](./gridworld_mdps_tutorial.ipynb) | MDPs en Gridworld: estados, acciones, recompensas, transiciones |
| [`policy_iteration_tutorial.ipynb`](./policy_iteration_tutorial.ipynb) | Algoritmo de Policy Iteration |
| [`value_iteration_tutorial.ipynb`](./value_iteration_tutorial.ipynb) | Algoritmo de Value Iteration |
| [`car.ipynb`](./car.ipynb) | MountainCar: formulación y exploración |
| [`value_car.ipynb`](./value_car.ipynb) | Value Iteration aplicado a MountainCar |
| [`policy_car.ipynb`](./policy_car.ipynb) | Policy Iteration aplicado a MountainCar |
| [`cross_lab2.ipynb`](./cross_lab2.ipynb) | Laboratorio cross-entropy y exploración |

## Orden de estudio recomendado

1. `tutorial_bandidos.ipynb` → `bandidos_epsilon.ipynb`
2. `gridworld_mdps_tutorial.ipynb`
3. `policy_iteration_tutorial.ipynb` → `value_iteration_tutorial.ipynb`
4. `car.ipynb` → `value_car.ipynb` → `policy_car.ipynb`
5. `cross_lab2.ipynb`

## Cómo ejecutar

```bash
pip install gymnasium numpy matplotlib
jupyter notebook
```
