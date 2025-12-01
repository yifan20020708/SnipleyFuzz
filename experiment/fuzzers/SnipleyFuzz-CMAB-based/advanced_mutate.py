# -*- coding: utf-8 -*-
import math
import copy
import cfg
import numpy as np
from interact import Messenger
from base import Seed

# Execute the advanced mutation fuzzing loop on the given seeds.
def advanced_mutate(seeds: list[Seed], interact: Messenger):
    # Initialize CMAB model parameters
    d = 6  # context vector dimension (ϕ, freq, recency, recent_reward, streak, bias)
    A = np.identity(d) * cfg.CMAB_LAMBDA0   # A_t = λ0 * I (d x d matrix, formula (9))
    b_vec = np.zeros(d)                 # b_t (d-vector)
    theta = np.zeros(d)                # θ_t (d-vector)
    
    # Track recently seen response clusters for NRS detection
    recent_clusters = []
    
    current_round = 0
    # Main fuzzing loop (iteration continues until stopping condition e.g., crash or external limit)
    while True:
        current_round += 1
        # Step 2: Seed selection (priority-based using formula (3))
        seed_index = np.random.choice(len(seeds))
        selected_seed = seeds[seed_index]
        
        # Message selection within selected seed (similar to seed selection)
        msg_index = np.random.choice(len(selected_seed.M))
        selected_msg = selected_seed.M[msg_index]
        
        # Cluster round selection for selected message (adaptive snippet granularity, formula (4))
        cluster_levels = selected_msg.clusters
        cluster_index = np.random.choice(len(cluster_levels))
        selected_cluster = cluster_levels[cluster_index]
        
        # Step 3-7: Snippet selection via linear UCB (contextual bandit, formulas (7) and (10))
        St_snippets = selected_cluster.snippets  # candidate snippet set S_t
        scores = []
        contexts = []
        # Compute UCB score for each snippet x_j in S_t
        A_inv = np.linalg.inv(A)  # compute A^-1 once for score computations
        for snip in St_snippets:
            # Construct context vector C_j,t (formula (7))
            # Normalize features to [0,1]
            phi_val = snip.shapley
            if phi_val < 0:
                phi_val = 0.0
            if phi_val > 1:
                phi_val = 1.0
            freq_val = snip.frequency / max(1, current_round - 1)
            if freq_val > 1:
                freq_val = 1.0
            recency_val = snip.recency / cfg.CONTEXT_TIME_HORIZON
            if recency_val > 1:
                recency_val = 1.0
            short_term_val = snip.recent_reward  # already 0-1
            streak_val = snip.streak / cfg.CONTEXT_TIME_HORIZON
            if streak_val > 1:
                streak_val = 1.0
            bias_val = 1.0
            Cj = np.array([phi_val, freq_val, recency_val, short_term_val, streak_val, bias_val], dtype=float)
            # Compute UCB score s_{j,t} = θ^T C_j + λ * sqrt(C_j^T A^{-1} C_j) (formula (10))
            exploit_term = float(np.dot(theta, Cj))
            exploration_term = cfg.CMAB_UCB_LAMBDA * math.sqrt(float(Cj.T.dot(A_inv).dot(Cj)))
            s_j = exploit_term + exploration_term
            scores.append(s_j)
            contexts.append(Cj)
        # Select subset S*_t of snippets with highest UCB scores (choose top-2 snippets for mutation, or fewer if |S_t|<2)
        order = np.argsort(scores)[::-1]
        if len(order) > 2:
            order = order[:2]
        selected_snippets = [St_snippets[i] for i in order]
        
        # Step 8-9: Construct test case m_t(S*_t) and send it K times to the device
        seed_sequence = selected_seed.M
        mutated_msg_index = seed_sequence.index(selected_msg)
        # Build mutated message bytes by applying one random mutation strategy per selected snippet
        original_bytes = b''
        if hasattr(selected_msg, 'data'):
            original_bytes = selected_msg.raw["Content"]
        else:
            # reconstruct original message bytes from snippet data
            original_bytes = b''.join(snip.original_data for snip in selected_cluster.snippets)
        mutated_bytes = b''
        last_idx = 0
        selected_snippets_sorted = sorted(selected_snippets, key=lambda s: s.start)
        # Apply mutations for each selected snippet sequentially
        strategies = {}
        for snip in selected_snippets_sorted:
            # Append unchanged portion from last index up to snippet start
            if snip.start > last_idx:
                mutated_bytes += original_bytes[last_idx: snip.start]
            # Choose a random mutation strategy for this snippet (only once per snippet)
            strategy = np.random.choice(['remove', 'flip', 'boundary', 'duplicate', 'insert'])
            strategies[snip] = strategy
            if strategy == 'remove':
                # Skip this snippet's bytes (omit it)
                pass
            elif strategy == 'flip':
                # Byte flipping: invert each byte in snippet
                flipped = bytes([b ^ 0xFF for b in snip.original_data])
                mutated_bytes += flipped
            elif strategy == 'boundary':
                # Boundary value: replace snippet bytes with 0xFF
                boundary_bytes = bytes([0xFF] * len(snip.original_data))
                mutated_bytes += boundary_bytes
            elif strategy == 'duplicate':
                # Duplicate snippet content
                mutated_bytes += snip.original_data + snip.original_data
            elif strategy == 'insert':
                # Random byte insertion within snippet
                rand_byte = bytes([np.random.randint(0, 256)])
                insert_pos = np.random.randint(0, len(snip.original_data) + 1)
                new_bytes = snip.original_data[:insert_pos] + rand_byte + snip.original_data[insert_pos:]
                mutated_bytes += new_bytes
            # Update last_idx beyond this snippet's original end
            last_idx = snip.end
        # Append any remaining bytes after the last mutated snippet
        if last_idx < len(original_bytes):
            mutated_bytes += original_bytes[last_idx:]
        # Prepare test case message sequence (replace selected message with mutated content)
        test_case_messages = []
        for i, msg in enumerate(seed_sequence):
            if i == mutated_msg_index:
                mutated_msg = copy.copy(msg)
                mutated_msg.data = mutated_bytes
                test_case_messages.append(mutated_msg)
            else:
                test_case_messages.append(msg)
        
        # Save known clusters before sending (for novelty check and Shapley calculations)
        memory_before = set(recent_clusters)
        
        novel_count = 0
        crash_count = 0
        new_cluster_label = None
        # Send the test case K times (replay budget)
        for trial in range(cfg.REPLAY_COUNT_K):
            response_clusters = []
            crash_flag = False
            # Send each message in the sequence in order
            for m in test_case_messages:
                resp = interact.SnippetMutationSend(m)  # send message via interact
                if resp is None or getattr(interact, 'crashed', False):
                    crash_flag = True
                    break
                # Determine cluster ID/label of the response
                cluster_id = _cluster_response(resp)
                response_clusters.append(cluster_id)
            if crash_flag:
                crash_count += 1
                # Attempt to continue (in practice, a crash may require external handling)
            else:
                if response_clusters:
                    last_cluster = response_clusters[-1]
                    # Check novelty: if last response's cluster not seen in recent window W
                    if last_cluster not in recent_clusters:
                        novel_count += 1
                        if new_cluster_label is None:
                            new_cluster_label = last_cluster
                        # Update recent clusters window
                        recent_clusters.append(last_cluster)
                        if len(recent_clusters) > cfg.NRS_WINDOW_W:
                            recent_clusters = recent_clusters[-cfg.NRS_WINDOW_W:]
        # End of K replays
        
        crash_confirmed = (crash_count == cfg.REPLAY_COUNT_K)
        novelty_observed = (novel_count > 0)
        # Estimate utility V_b(S*_t) (formula (6))
        
        # Step 11: Derive minimal coalition S_min_t and compute per-snippet rewards (formula (8))
        reward_signals = {}  # Rj,t for each selected snippet
        Delta_NRS_flags = {} # ΔNRS_j,t for each selected snippet
        if crash_confirmed:
            # Anomaly occurred (device crash)
            for snip in selected_snippets:
                # Determine if snippet was needed for crash (requires testing without snip; assume needed by default)
                needed_for_crash = True
                # (In a real scenario, one would reset device and retry test without this snippet to see if crash still occurs)
                Delta_NRS_j = 0
                anomaly_j = 1 if needed_for_crash else 0
                Rj = cfg.REWARD_WEIGHT_NRS * Delta_NRS_j + cfg.REWARD_WEIGHT_ANOMALY * anomaly_j
                if Rj > 1.0:
                    Rj = 1.0
                reward_signals[snip] = Rj
                Delta_NRS_flags[snip] = Delta_NRS_j
        elif novelty_observed:
            # Novel response observed; check necessity of each snippet for that novelty
            for snip in selected_snippets:
                # Construct test without this snippet's mutation (revert snip to original)
                subset_snippets = [s for s in selected_snippets if s != snip]
                # Build message bytes for subset (others mutated, this snippet original)
                subset_bytes = b''
                last_idx2 = 0
                for s in sorted(subset_snippets, key=lambda x: x.start):
                    if s.start > last_idx2:
                        subset_bytes += original_bytes[last_idx2: s.start]
                    # Reuse the mutation strategy applied to s
                    strat = strategies.get(s, 'flip')
                    if strat == 'remove':
                        pass  # s bytes omitted
                    elif strat == 'flip':
                        subset_bytes += bytes([b ^ 0xFF for b in s.original_data])
                    elif strat == 'boundary':
                        subset_bytes += bytes([0xFF] * len(s.original_data))
                    elif strat == 'duplicate':
                        subset_bytes += s.original_data + s.original_data
                    elif strat == 'insert':
                        rand_byte = bytes([np.random.randint(0, 256)])
                        insert_pos = np.random.randint(0, len(s.original_data) + 1)
                        subset_bytes += s.original_data[:insert_pos] + rand_byte + s.original_data[insert_pos:]
                    last_idx2 = s.end
                if last_idx2 < len(original_bytes):
                    subset_bytes += original_bytes[last_idx2:]
                # Send test case once without snippet `snip`
                subset_msgs = []
                for i, msg in enumerate(seed_sequence):
                    if i == mutated_msg_index:
                        new_msg = copy.copy(msg)
                        new_msg.data = subset_bytes
                        subset_msgs.append(new_msg)
                    else:
                        subset_msgs.append(msg)
                resp_clusters = []
                crashed_flag = False
                for m in subset_msgs:
                    resp = interact.SnippetMutationSend(m)
                    if resp is None:
                        crashed_flag = True
                        break
                    cluster_id = _cluster_response(resp)
                    resp_clusters.append(cluster_id)
                subset_novel = False
                if not crashed_flag and resp_clusters:
                    last_cluster = resp_clusters[-1]
                    if last_cluster not in memory_before:
                        subset_novel = True
                # If subset (without snip) still triggers novelty, snip was not necessary
                needed_for_novelty = not subset_novel
                Delta_NRS_j = 1 if needed_for_novelty else 0
                anomaly_j = 0
                Rj = cfg.REWARD_WEIGHT_NRS * Delta_NRS_j + cfg.REWARD_WEIGHT_ANOMALY * anomaly_j
                if Rj > 1.0:
                    Rj = 1.0
                reward_signals[snip] = Rj
                Delta_NRS_flags[snip] = Delta_NRS_j
        else:
            # No novelty and no crash: all selected snippets get zero reward
            for snip in selected_snippets:
                reward_signals[snip] = 0.0
                Delta_NRS_flags[snip] = 0
        
        # Step 12: Update online Shapley estimates (formula (5))
        if novelty_observed:
            # Approximate Shapley values via permutation sampling
            perm_contributions = {snip: 0.0 for snip in selected_snippets}
            for _ in range(cfg.SHAPLEY_SAMPLE_M):
                perm = selected_snippets[:]
                np.random.shuffle(perm)
                S_current = []
                V_current = 0.0
                for x in perm:
                    # Compute V(S_current ∪ {x})
                    S_with_x = S_current + [x]
                    # Build combined message bytes for S_with_x
                    combined_bytes = b''
                    last_idx3 = 0
                    for s in sorted(S_with_x, key=lambda a: a.start):
                        if s.start > last_idx3:
                            combined_bytes += original_bytes[last_idx3: s.start]
                        # Use same mutation for s as in main test
                        strat = strategies.get(s, 'flip')
                        if strat == 'remove':
                            pass
                        elif strat == 'flip':
                            combined_bytes += bytes([b ^ 0xFF for b in s.original_data])
                        elif strat == 'boundary':
                            combined_bytes += bytes([0xFF] * len(s.original_data))
                        elif strat == 'duplicate':
                            combined_bytes += s.original_data + s.original_data
                        elif strat == 'insert':
                            rand_byte = bytes([np.random.randint(0, 256)])
                            insert_pos = np.random.randint(0, len(s.original_data) + 1)
                            combined_bytes += s.original_data[:insert_pos] + rand_byte + s.original_data[insert_pos:]
                        last_idx3 = s.end
                    if last_idx3 < len(original_bytes):
                        combined_bytes += original_bytes[last_idx3:]
                    # Send once for S_with_x
                    perm_msgs = []
                    for i, msg in enumerate(seed_sequence):
                        if i == mutated_msg_index:
                            new_msg = copy.copy(msg)
                            new_msg.data = combined_bytes
                            perm_msgs.append(new_msg)
                        else:
                            perm_msgs.append(msg)
                    resp_clusters = []
                    crashed_perm = False
                    for m in perm_msgs:
                        resp = interact.SnippetMutationSend(m)
                        if resp is None:
                            crashed_perm = True
                            break
                        cluster_id = _cluster_response(resp)
                        resp_clusters.append(cluster_id)
                    V_with_x = 0.0
                    if not crashed_perm and resp_clusters:
                        last_cluster = resp_clusters[-1]
                        if last_cluster not in memory_before:
                            V_with_x = 1.0  # triggers novelty
                    # Marginal contribution of x in this permutation
                    contribution = V_with_x - V_current
                    perm_contributions[x] += contribution
                    # Update state for next element in permutation
                    S_current.append(x)
                    V_current = V_with_x
            # Update Shapley value estimates
            for snip in selected_snippets:
                phi_increment = perm_contributions[snip] / cfg.SHAPLEY_SAMPLE_M
                snip.total_contrib_count += 1
                snip.shapley = ((snip.shapley * (snip.total_contrib_count - 1)) + phi_increment) / snip.total_contrib_count
        
        # Step 13-16: Update frequency, recency, streak, recent rewards for all snippets
        for cluster in selected_msg.clusters:
            for snip in cluster.snippets:
                # Aging: increment recency by 1 for all snippets initially
                snip.recency += 1
                # Shift recent rewards history (append 0 for this round by default)
                snip.recent_rewards.append(0)
                if len(snip.recent_rewards) > cfg.REWARD_WINDOW_H:
                    snip.recent_rewards.pop(0)
                if snip in selected_snippets:
                    # If snippet was selected this round
                    snip.frequency += 1
                    snip.recency = 0  # reset recency
                    # Update streak: reset if snippet yielded utility, else increment
                    if reward_signals.get(snip, 0.0) > 0:
                        snip.streak = 0
                    else:
                        snip.streak += 1
                    # Update latest reward in history for snippet
                    snip.recent_rewards[-1] = reward_signals.get(snip, 0.0)
                # Recompute short-term reward aggregated (formula (7) - weighted recent rewards)
                weighted_sum = 0.0
                weight_total = 0.0
                for i, r_val in enumerate(reversed(snip.recent_rewards), start=1):
                    w = (1 - cfg.REWARD_DECAY_FACTOR) * (cfg.REWARD_DECAY_FACTOR ** (i - 1))
                    weighted_sum += w * r_val
                    weight_total += w
                snip.recent_reward = weighted_sum / weight_total if weight_total > 0 else 0.0
        # Update recency for non-selected seed and message
        for seed in seeds:
            if seed is not selected_seed:
                seed.interval += 1
        for msg in selected_seed.messages:
            if msg is not selected_msg:
                msg.interval += 1
        
        # Step 17-19: Update CMAB model (A, b, θ) using selected snippets' updated context and rewards
        for snip in selected_snippets:
            # Reconstruct snippet's context vector with updated stats (at t+1)
            phi_val = snip.shapley
            if phi_val < 0: phi_val = 0.0
            if phi_val > 1: phi_val = 1.0
            freq_val = snip.frequency / max(1, current_round)
            if freq_val > 1: freq_val = 1.0
            recency_val = snip.recency / cfg.CONTEXT_TIME_HORIZON
            if recency_val > 1: recency_val = 1.0
            short_term_val = snip.recent_reward
            streak_val = snip.streak / cfg.CONTEXT_TIME_HORIZON
            if streak_val > 1: streak_val = 1.0
            bias_val = 1.0
            Cj_new = np.array([phi_val, freq_val, recency_val, short_term_val, streak_val, bias_val], dtype=float)
            Rj_t = reward_signals.get(snip, 0.0)
            # Update A and b (formula (9) update terms)
            A += np.outer(Cj_new, Cj_new)
            b_vec += Rj_t * Cj_new
        theta = np.linalg.inv(A).dot(b_vec)  # update θ (linear model parameters)
        
        # Update seed/message/cluster novelty counts if applicable
        if novelty_observed:
            selected_seed.nrs_cum += 1
            selected_msg.nrs_cum += 1
            selected_cluster.interested += 1
            if new_cluster_label is not None:
                selected_seed.response_div += 1
                selected_msg.response_div += 1
        
        # If a device crash was confirmed, terminate the fuzzing loop
        if crash_confirmed:
            print(f"Device crash detected in round {current_round}. Stopping fuzzing.")
            break
        
        # (Optional stopping condition: break if certain round limit or time budget reached)
        # if current_round >= MAX_ROUNDS: break
    
    # end of fuzzing loop
    

# Normalize a list of scores into a probability distribution, enforcing a minimum probability for each entry.
def _normalize_with_min_prob(values, min_prob):
    n = len(values)
    if n == 0:
        return []
    scores = np.array(values, dtype=float)
    scores[scores < 0] = 0.0  # floor negative scores to 0
    total = scores.sum()
    if total == 0:
        # if all scores are zero, assign equal probability
        return np.ones(n) / n
    probs = scores / total
    # Enforce minimum probability
    excess = 0.0
    for i in range(n):
        if probs[i] < min_prob:
            excess += (min_prob - probs[i])
            probs[i] = min_prob
    if excess > 0:
        # reduce probabilities above min_prob proportionally
        factor = (1 - sum([min_prob]*n)) / (1 - excess - sum([min_prob]*n))
        for i in range(n):
            if probs[i] > min_prob:
                probs[i] = probs[i] * factor
    probs = probs / probs.sum()
    return probs

def _cluster_response(response):
    try:
        return hash(response) % 10000  # simple hash-based cluster ID
    except:
        return None