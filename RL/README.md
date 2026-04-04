# 🎮 RL — Reinforcement Learning

Módulo de **Aprendizaje por Refuerzo** del programa MAIA. Cubre los fundamentos teóricos y prácticos del RL: desde bandidos multi-brazo hasta Q-Learning y proyectos completos de agentes inteligentes.

---

## 📚 Contenido

| Recurso | Descripción |
|---------|-------------|
| [notebooks/](./notebooks/) | Laboratorios y tutoriales: bandidos, MDPs, policy/value iteration |
| [Proyecto/](./Proyecto/) | Proyecto final: agente RL para resolución de laberintos |
| [AI-Pacman-Projects-master/](./AI-Pacman-Projects-master/) | Proyectos UC Berkeley de AI con Pac-Man |

---

## 🎯 Objetivos de aprendizaje

- Formalizar problemas de toma de decisiones como MDPs (procesos de decisión de Markov)
- Implementar y comparar políticas de exploración: ε-greedy, UCB, Thompson Sampling
- Aplicar algoritmos de planificación: Policy Iteration y Value Iteration
- Implementar Q-Learning y SARSA en entornos discretos y continuos
- Entrenar agentes en entornos OpenAI Gym (CartPole, MountainCar)
- Desarrollar un proyecto completo de agente para resolución de laberintos

---

## 🚀 Configuración del entorno

```bash
pip install gymnasium numpy matplotlib
pip install torch  # Para DQN y extensiones

jupyter notebook
```

---

## 🔗 Referencias

- [Sutton & Barto — Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)
- [OpenAI Gymnasium](https://gymnasium.farama.org/)
- [David Silver's RL Course (DeepMind)](https://www.deepmind.com/learning-resources/introduction-to-reinforcement-learning-with-david-silver)
- [Berkeley AI Pac-Man Projects](https://ai.berkeley.edu/project_overview.html)
