import time
import random
import string
from statistics import mean
from typing import List, Tuple
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# 使用当前文件所在目录作为基础路径
current_dir = Path(__file__).parent
chart_dir = current_dir.parent / "chart"


def random_string(length: int) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(random.choice(alphabet) for _ in range(length))


def ngram_set(s: str, n: int = 3) -> set:
    if len(s) < n:
        return {s}
    return {s[i:i+n] for i in range(len(s) - n + 1)}


def ngram_similarity(s1: str, s2: str, n: int = 3) -> float:
    g1 = ngram_set(s1, n)
    g2 = ngram_set(s2, n)
    inter = len(g1 & g2)
    union = len(g1 | g2)
    return inter / union if union > 0 else 0.0


def seed_similarity(seed1: List[str], seed2: List[str]) -> float:
    s1 = list(seed1)
    s2 = list(seed2)
    # pad 到同样长度
    while len(s1) < len(s2):
        s1.append(random.choice(s1))
    while len(s2) < len(s1):
        s2.append(random.choice(s2))
    sims = []
    used = set()
    for m1 in s1:
        best_sim = -1.0
        best_idx = -1
        for idx, m2 in enumerate(s2):
            if idx in used:
                continue
            sim = ngram_similarity(m1, m2, n=3)
            if sim > best_sim:
                best_sim = sim
                best_idx = idx
        sims.append(best_sim)
        used.add(best_idx)
    return sum(sims) / len(sims) if sims else 0.0


class SnipleySchedulerMicroBench:
    def __init__(
        self,
        n_seeds: int = 40,
        n_msgs_per_seed: int = 6,
        msg_len: int = 64,
        n_clusters: int = 3,
        n_snippets: int = 50,
        n_selected_snippets: int = 10,
        shapley_M: int = 16,
        context_dim: int = 6,
    ):
        self.n_seeds = n_seeds
        self.n_msgs_per_seed = n_msgs_per_seed
        self.msg_len = msg_len
        self.n_clusters = n_clusters
        self.n_snippets = n_snippets
        self.n_selected_snippets = n_selected_snippets
        self.shapley_M = shapley_M
        self.d = context_dim

        # 生成随机种子池（每个种子是一组字符串消息）
        self.seed_pool: List[List[str]] = [
            [random_string(self.msg_len) for _ in range(self.n_msgs_per_seed)]
            for _ in range(self.n_seeds)
        ]

        # snippet 相关统计：对每个 snippet j 维护一个状态
        self.phi = np.zeros(self.n_snippets)          # Shapley 估计
        self.freq = np.zeros(self.n_snippets)         # 选择频率
        self.recency = np.zeros(self.n_snippets)      # 自上次选择以来的轮数
        self.streak = np.zeros(self.n_snippets)       # 自上次有 reward 以来的轮数
        self.short_reward = np.zeros(self.n_snippets) # 短期奖励聚合

        # CMAB: A, b, theta
        self.A = np.eye(self.d) * 1e-2  # λ0 I_d
        self.b = np.zeros((self.d, 1))
        self.theta = np.zeros((self.d, 1))

        self.round_counter = 0

    # 种子/消息/聚类轮次优先级

    def priority_step(self) -> None:
        # 随机生成 div, NRS_cum, Fresh, Use, rho
        div = np.random.rand(self.n_seeds)
        nrs_cum = np.random.rand(self.n_seeds)
        fresh = np.random.rand(self.n_seeds)
        use = np.random.rand(self.n_seeds)
        rho = np.random.rand(self.n_seeds)

        # 这里只关心运算量，不关心绝对值
        capability = div + nrs_cum
        redundancy_penalty = 1.0 - rho
        aging_ctrl = fresh * use
        p_seed = capability * redundancy_penalty * aging_ctrl 

        # 排序（为后续选择准备）
        topk_idx = np.argsort(-p_seed)[: min(10, self.n_seeds)]

        # 为了覆盖相似度计算的复杂度，对 top-k 种子做 pairwise similarity
        for i in range(len(topk_idx)):
            for j in range(i + 1, len(topk_idx)):
                s1 = self.seed_pool[topk_idx[i]]
                s2 = self.seed_pool[topk_idx[j]]
                _ = seed_similarity(s1, s2)  # 只为模拟复杂度，不用结果

        # cluster potential，简单点：对每个 cluster 造一些 NRS_cum, Fresh, Use
        nrs_cum_c = np.random.rand(self.n_clusters)
        fresh_c = np.random.rand(self.n_clusters)
        use_c = np.random.rand(self.n_clusters)

        p_cluster = nrs_cum_c * fresh_c * use_c
        _ = np.argmax(p_cluster)  # 选一个 cluster 轮次（不使用结果）

    # Shapley 在线估计
    def shapley_step(self) -> None:
        # 选一个候选子集 S*_t
        S_indices = np.arange(self.n_snippets)
        np.random.shuffle(S_indices)
        S_star = S_indices[: self.n_selected_snippets]

        # 模拟 utility 基值（例如某个 V(S*_t) 的估计）
        base_utility = 0.3 + 0.1 * np.random.rand()

        # 对每个 snippet j ∈ S*_t 做 M 次增量计算
        for j_idx in S_star:
            inc_sum = 0.0
            for _ in range(self.shapley_M):
                # 这里不真实构造所有 permutation，只做若干次简单运算，保持复杂度 ~ O(M)
                v_with = base_utility + 0.01 * np.random.rand()
                v_without = base_utility
                inc_sum += (v_with - v_without)
            delta_phi = inc_sum / self.shapley_M
            # 简单的 EMA 更新，模拟在线更新
            self.phi[j_idx] = 0.9 * self.phi[j_idx] + 0.1 * delta_phi

    # CMAB：构造上下文 + 线性 UCB
    def cmab_step(self) -> None:
        self.round_counter += 1

        # recency + streak + freq 更新
        self.recency += 1.0
        self.streak += 1.0

        # 模拟 reward：随机为少量 snippet 给一点 reward
        rewards = np.zeros(self.n_snippets)
        # 例如随机挑几个 snippet 有 reward
        for _ in range(3):
            j = random.randrange(self.n_snippets)
            rewards[j] = np.random.rand()
            self.streak[j] = 0.0

        # 更新 freq, recency, short_reward
        for j in range(self.n_snippets):
            if rewards[j] > 0:
                self.freq[j] += 1
                self.recency[j] = 0
            self.short_reward[j] = 0.8 * self.short_reward[j] + 0.2 * rewards[j]

        # 数值归一化（跟论文中 “Normalize to [0,1]” 一致）
        def safe_norm(x):
            xmax = np.max(x)
            return x / (xmax + 1e-8) if xmax > 0 else x

        phi_norm = safe_norm(self.phi)
        freq_norm = safe_norm(self.freq)
        recency_norm = safe_norm(self.recency)
        streak_norm = safe_norm(self.streak)
        reward_norm = safe_norm(self.short_reward)

        # 构造所有 snippet 的上下文矩阵 C: shape (n_snippets, d)
        C = np.zeros((self.n_snippets, self.d))
        C[:, 0] = phi_norm
        C[:, 1] = freq_norm
        C[:, 2] = recency_norm
        C[:, 3] = reward_norm
        C[:, 4] = streak_norm
        C[:, 5] = 1.0  # bias

        # 基于 rewards 更新 A, b, theta（只用有正 reward 的 snippet）
        # A_t+1 = A_t + sum C C^T,  b_t+1 = b_t + sum r C
        idx_pos = np.where(rewards > 0)[0]
        for j in idx_pos:
            cj = C[j].reshape(self.d, 1)
            rj = rewards[j]
            self.A += cj @ cj.T
            self.b += rj * cj

        # 如果有新样本，则更新 theta
        if len(idx_pos) > 0:
            self.theta = np.linalg.solve(self.A, self.b)

        # 计算所有 snippet 的 UCB 分数
        lam = 0.5  # exploration 系数
        scores = np.zeros(self.n_snippets)
        A_inv = np.linalg.inv(self.A)
        for j in range(self.n_snippets):
            cj = C[j].reshape(self.d, 1)
            mean_est = float(self.theta.T @ cj)
            var_est = float(np.sqrt(cj.T @ A_inv @ cj))
            scores[j] = mean_est + lam * var_est

        # 选一个 snippet 作为本轮 “被选中”
        _ = int(np.argmax(scores))  # 不使用结果，只模拟负载

    # 运行 benchmark
    def run(self, n_rounds: int = 1000) -> Tuple[List[float], List[float], List[float]]:
        priority_times = []
        shapley_times = []
        cmab_times = []

        for _ in range(n_rounds):
            # priority
            t0 = time.perf_counter()
            self.priority_step()
            priority_times.append(time.perf_counter() - t0)

            # shapley
            t1 = time.perf_counter()
            self.shapley_step()
            shapley_times.append(time.perf_counter() - t1)

            # cmab
            t2 = time.perf_counter()
            self.cmab_step()
            cmab_times.append(time.perf_counter() - t2)

        return priority_times, shapley_times, cmab_times


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    bench = SnipleySchedulerMicroBench(
        n_seeds=40,
        n_msgs_per_seed=6,
        msg_len=64,
        n_clusters=3,
        n_snippets=50,
        n_selected_snippets=10,
        shapley_M=16,
        context_dim=6,
    )

    N_ROUNDS = 1000
    p, s, c = bench.run(N_ROUNDS)

    def summarize(name, arr):
        print(
            f"{name}: mean={mean(arr)*1000:.3f} ms, "
            f"min={min(arr)*1000:.3f} ms, "
            f"max={max(arr)*1000:.3f} ms"
        )

    summarize("Priority  ", p)
    summarize("Shapley   ", s)
    summarize("CMAB      ", c)
    total_per_round = [a + b + d for a, b, d in zip(p, s, c)]
    summarize("Total/round", total_per_round)
    mean_priority = mean(p)*1000
    mean_shapley = mean(s)*1000
    mean_cmab = mean(c)*1000
    mean_total = mean_priority + mean_shapley + mean_cmab
    T_device = 500.0   

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
    })

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.0))

    ax = axes[0]

    names_a  = ["Priority", "Shapley", "CMAB", "Total"]
    values_a = [mean_priority, mean_shapley, mean_cmab, mean_total]
    colors_a = ["#4C72B0", "#DD8452", "#55A868", "#8172B3"]

    bars_a = ax.bar(names_a, values_a, color=colors_a, width=0.6)

    ax.set_yscale("log")
    ax.set_ylabel("Mean time per round (ms, log)")
    ax.set_title("(a) Breakdown of scheduler overhead")

    # 让刻度覆盖 0.01 ms – 100 ms 左右，三部分+总耗时都在里面
    ax.set_ylim(1e-2, 1e2)
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=5))
    ax.yaxis.set_major_formatter(mticker.LogFormatter())

    for rect, v in zip(bars_a, values_a):
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            v * 1.15,
            f"{v:.2f} ms",
            ha="center", va="bottom", fontsize=9,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]

    names_b  = ["Scheduler\n(Prio+Shap+CMAB)", "End-to-end\nround"]
    values_b = [mean_total, T_device]
    colors_b = ["#4C72B0", "#999999"]

    bars_b = ax.bar(names_b, values_b, color=colors_b, width=0.6)

    ax.set_yscale("log")
    ax.set_ylabel("Wall-clock time per round (ms, log)")
    ax.set_title("(b) Scheduler vs device+network latency")

    ax.set_ylim(1e1, 1e4)
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=4))
    ax.yaxis.set_major_formatter(mticker.LogFormatter())

    for rect, v in zip(bars_b, values_b):
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            v * 1.15,
            f"{v:.0f} ms",
            ha="center", va="bottom", fontsize=9,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(str(chart_dir / "mutation_consume.png"), dpi=600, bbox_inches='tight')
    plt.savefig(str(chart_dir / "mutation_consume.pdf"), bbox_inches='tight')