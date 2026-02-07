# Adversarial Fraud Detection

## Overview

This project implements an adversarial fraud detection system where a Q-learning agent learns to evade fraud detection, and the model adapts to these adversarial attacks.

## Adversarial Training Process

### 1. Base Model Training
Train initial XGBoost model on IEEE-CIS dataset:
```bash
python train.py
```

### 2. Adversarial Agent Training
Q-learning agent learns to evade detection:
```bash
python train_adversarial.py
```

The agent:
- Observes transaction features (state)
- Chooses manipulations (actions): increase/decrease amount, change timing, split transactions
- Receives rewards: +1 if fraud undetected, -1 if detected
- Updates Q-values using Q-learning

### 3. Model Adaptation
In production, adversarial examples can be used to:
- Retrain model on harder examples
- Improve robustness against adaptive fraudsters
- Test model resilience

## Q-Learning Details

**State Space**: Discretized transaction features (amount bins, time bins)

**Action Space**:
- `increase_amount`: Multiply amount by 1.1-1.5x
- `decrease_amount`: Multiply amount by 0.5-0.9x
- `change_time_early`: Shift to morning hours (6-12)
- `change_time_late`: Shift to evening hours (18-23)
- `split_transaction`: Divide into 2-4 smaller transactions

**Reward Function**:
- +1: Fraud not detected (successful evasion)
- -1: Fraud detected (failed evasion)

**Learning Parameters**:
- Learning rate (α): 0.1
- Discount factor (γ): 0.95
- Exploration rate (ε): 0.1

## Results

The adversarial training loop shows:
- Agent learns to evade detection over episodes
- Evasion rate increases as Q-table grows
- Model can be retrained on adversarial examples to improve robustness

## Interview Talking Points

1. **Why adversarial training?**
   - Real fraudsters adapt to detection systems
   - Adversarial training makes models more robust
   - Demonstrates understanding of ML security

2. **Why Q-learning?**
   - Model-free RL suitable for discrete actions
   - Learns optimal evasion strategy
   - Computationally efficient for this problem

3. **Production deployment:**
   - Run agent in shadow mode to generate test cases
   - Periodically retrain on adversarial examples
   - Monitor evasion rate as model health metric
