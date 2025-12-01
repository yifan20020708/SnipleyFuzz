# -*- coding: utf-8 -*-

# Seed/message priority weights (formula (3) components)
ALPHA_CAPABILITY_SEED = 1.0    # weight α for seed capability (diversity term)
BETA_CAPABILITY_SEED = 1.0     # weight β for seed capability (cumulative NRS term)
ALPHA_CAPABILITY_MSG = 1.0     # weight α for message capability
BETA_CAPABILITY_MSG = 1.0      # weight β for message capability
ALPHA_CAPABILITY_CLUSTER = 1.0 # weight α for cluster potential (novelty term, formula (4))

# Redundancy and aging weights (for Fresh/Use factors)
FRESH_WEIGHT_SEED = 1.0        # weight for Fresh(seed) factor
USE_WEIGHT_SEED = 1.0          # weight for Use(seed) factor
FRESH_WEIGHT_MSG = 1.0         # weight for Fresh(message) factor
USE_WEIGHT_MSG = 1.0           # weight for Use(message) factor
FRESH_WEIGHT_CLUSTER = 1.0     # weight for Fresh(cluster) factor
USE_WEIGHT_CLUSTER = 1.0       # weight for Use(cluster) factor

# Minimum selection probability to avoid starvation
LEAST_PROB_SEED = 0.05         # minimum probability for any seed to be selected
LEAST_PROB_MSG = 0.05          # minimum probability for any message to be selected
LEAST_PROB_CLUSTER = 0.05      # minimum probability for any cluster level to be selected

# Shapley estimation parameters
SHAPLEY_SAMPLE_M = 16          # number of random permutations M for Shapley approximation (formula (5))
SHAPLEY_DELTA_VALUE = 1.0      # base unit for marginal contribution increment (use 1.0 for normalization)

# Novel Response State (NRS) detection window and replay count
NRS_WINDOW_W = 180            # time window W (in fuzzing rounds) for considering a response "novel" (formula (2))
REPLAY_COUNT_K = 3            # replay budget K (number of replays per test case, formula (6))

# Reward signal weights (formula (8))
REWARD_WEIGHT_NRS = 1.0        # α weight for NRS component of reward
REWARD_WEIGHT_ANOMALY = 2.0    # γ weight for anomaly (crash) component of reward

# Contextual Multi-Armed Bandit (CMAB) parameters
CMAB_LAMBDA0 = 1e-2            # ridge regression regularization coefficient λ0 (for A matrix initialization, formula (9))
CMAB_UCB_LAMBDA = 0.5          # UCB exploration coefficient λ (controls exploration intensity in formula (10))
REWARD_WINDOW_H = 3            # short-term reward window h (number of recent rounds to aggregate rewards, formula (7))
REWARD_DECAY_FACTOR = 0.5      # exponential decay factor (λ in formula (7)) for recent reward aggregation
CONTEXT_TIME_HORIZON = 10      # finite horizon for normalizing time-dependent features (e.g., recency, streak)

# Timeout and retry settings for device communication
YEELIGHT_TIMEOUT_TIMES = 2
YEELIGHT_MAX_RETRY = 3
XIAOMI_TIMEOUT_TIMES = 5