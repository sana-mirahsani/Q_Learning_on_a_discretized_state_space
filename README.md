# 🚀 Q-Learning on a Discretized State Space

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Reinforcement Learning](https://img.shields.io/badge/RL-Q--Learning-orange.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📌 Description

Welcome to the **Q-Learning on a Discretized State Space** project!

This project combines:

* 🔹 **Reinforcement Learning (Q-Learning)**
* 🔹 **Discretization of Continuous State Spaces**

It demonstrates how to train an agent in a **continuous environment** by:

1. Discretizing the state space (position & velocity)
2. Applying **Q-Learning** on the discrete representation
3. Learning optimal behavior through interaction

🎯 Example implemented:

* Inverted Pendulum Control

---

## 📂 Project Structure

```bash
project/
│
├── Environment.py
├── agent.py
├── run_program.py
```

| File             | Description                                                             |
| ---------------- | ----------------------------------------------------------------------- |
| `Environment.py` | Defines the continuous environment (transition, reward, stop condition) |
| `agent.py`       | Implements the Q-Learning agent with discretized state space            |
| `run_program.py` | Runs the full experiment (pendulum problem)                             |

---

## ✨ Features

### 🔹 Continuous Environment

* Custom environment class:

  * Transition function
  * Reward function
  * Stop condition
* Supports:

  * Deterministic environments
  * Continuous state representation (p, v)

---

### 🔹 State Space Discretization

* Converts continuous space into a **grid-based representation**

* Defines:

  * Position bins
  * Velocity bins
  * Grid cells

* Utilities:

  * Continuous → Discrete (`find_cell`)
  * Discrete → Continuous (`find_p_v`)

---

### 🔹 Q-Learning Algorithm

* Implements **tabular Q-Learning**:

  * Q(s, a) updates using TD learning

* Supports:

  * ✅ Epsilon-Greedy exploration
  * ✅ Boltzmann (Softmax) exploration

* Key components:

  * Learning rate:

    * Fixed (deterministic env)
    * Adaptive (stochastic env)
  * Discount factor γ
  * Episode-based training

---

### 🔹 Exploration Strategies

#### 🎯 Epsilon-Greedy

* Random exploration with probability ε
* Greedy exploitation otherwise

#### 🔥 Boltzmann (Softmax)

* Probabilistic action selection
* Controlled by temperature T

---

### 🔹 Training Process

* Multiple episodes (`EPISODE_BLOCK`)

* Episode termination:

  * Environment condition
  * Step limit (500 steps)

* Tracks:

  * Q-table updates
  * Epsilon decay
  * Total interactions

---

### 🔹 Example: Pendulum Control

* Physics-based continuous system
* Torque actions: `[-5, 0, 5]`
* Goal:

  * Stabilize pendulum upright

---

## ⚙️ Requirements

* Python 3.x
* NumPy

Install dependencies:

```bash
pip install numpy
```

---

## ▶️ How to Use

### 1. Clone the repository

```bash
git clone https://github.com/sana-mirahsani/Q_Learning_on_discretized_state_space
cd Q_Learning_on_discretized_state_space
```

### 2. Run the program

```bash
python run_program.py
```

---

## 📊 Example Output

After training, the program prints:

* Q-table shape
* Minimum Q-value
* Maximum Q-value
* Mean Q-value

Example:

```
Q-table shape: (400, 3)
Min Q-value: -0.5321
Max Q-value: 1.2345
Mean Q-value: 0.1023
```

---

## 💡 Key Concepts

* Q-Learning (Off-policy RL)
* Temporal Difference Learning
* Continuous → Discrete approximation
* Exploration vs Exploitation trade-off

---

## 🔧 Customization

You can easily extend this project:

* 🔁 Change environment:

  * Modify transition & reward functions
* 🔢 Adjust discretization:

  * Number of grid cells
* 🎯 Try different policies:

  * Switch between epsilon-greedy & Boltzmann

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 👩‍💻 Author

**Sana Mirahsani**
📧 [s.mirahsani1998@gmail.com](mailto:s.mirahsani1998@gmail.com)
🔗 LinkedIn: sana-mirahsani
💻 GitHub: sana-mirahsani

---

## ⭐ Support

If you find this project useful:

* ⭐ Star the repo
* 🍴 Fork it
* 🚀 Use it in your RL projects