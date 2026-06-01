from __future__ import annotations

import csv
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev


SEED = 20260514
TRAIN_EPISODES = int(os.environ.get("BAYES_SAFE_TRAIN_EPISODES", "40000"))
EVAL_EPISODES = int(os.environ.get("BAYES_SAFE_EVAL_EPISODES", "2000"))
MAX_ROUNDS = 10
TARGET_UTILITY = 0.70
CVaR_ALPHA = 0.10
CVaR_SAFE_FLOOR = 0.68
DISAGREEMENT_UTILITY = 0.20
OUT_DIR = Path("_bayesian_safe_rl_compare")
OUT_DIR.mkdir(exist_ok=True)


@dataclass(frozen=True)
class Issue:
    name: str
    low: float
    high: float
    buyer_weight: float
    seller_weight: float
    buyer_pref: str
    seller_pref: str


ISSUES = [
    Issue("unit_price_twd", 75, 500, 0.35, 0.35, "lower", "higher"),
    Issue("quantity_units", 10_000, 50_000, 0.25, 0.25, "lower", "higher"),
    Issue("delivery_weeks", 1, 5, 0.20, 0.10, "lower", "higher"),
    Issue("payment_months", 1, 5, 0.10, 0.10, "higher", "lower"),
    Issue("contract_years", 1, 5, 0.10, 0.20, "lower", "higher"),
]

ROLE_WEIGHTS = {
    "buyer": [i.buyer_weight for i in ISSUES],
    "seller": [i.seller_weight for i in ISSUES],
}
ROLE_PREFS = {
    "buyer": [i.buyer_pref for i in ISSUES],
    "seller": [i.seller_pref for i in ISSUES],
}


@dataclass(frozen=True)
class OpponentType:
    name: str
    concession_mean: float
    concession_sd: float
    agreement_bias: float
    pressure: float
    volatility: float
    synergy: float


OPPONENT_TYPES = [
    OpponentType("cooperative", 0.070, 0.025, 0.24, 0.30, 0.035, 0.12),
    OpponentType("balanced", 0.040, 0.030, 0.12, 0.55, 0.045, 0.08),
    OpponentType("pressure_sensitive", 0.085, 0.055, 0.30, 0.82, 0.070, 0.06),
    OpponentType("inconsistent", 0.048, 0.090, 0.16, 0.68, 0.130, 0.04),
    OpponentType("hardball", 0.012, 0.026, 0.05, 0.90, 0.035, 0.02),
]

HOLD = 0
SMALL = 1
MEDIUM = 2
TRADE = 3
STOP = 4
ACTIONS = [HOLD, SMALL, MEDIUM, TRADE, STOP]
ACTION_NAMES = {
    HOLD: "hold",
    SMALL: "small_concession",
    MEDIUM: "medium_concession",
    TRADE: "integrative_trade",
    STOP: "stop",
}


def clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def score_issue(z: float, pref: str) -> float:
    return z if pref == "higher" else 1.0 - z


def utility(role: str, offer: list[float]) -> float:
    return sum(
        w * score_issue(z, pref)
        for w, z, pref in zip(ROLE_WEIGHTS[role], offer, ROLE_PREFS[role])
    )


def concrete_offer(offer: list[float]) -> dict[str, float]:
    return {issue.name: issue.low + z * (issue.high - issue.low) for issue, z in zip(ISSUES, offer)}


def mandate_ok(role: str, offer: list[float]) -> bool:
    values = concrete_offer(offer)
    if role == "buyer":
        return (
            values["unit_price_twd"] <= 500
            and values["quantity_units"] <= 50_000
            and values["delivery_weeks"] <= 5
            and values["payment_months"] >= 1
            and values["contract_years"] >= 1
        )
    return (
        values["unit_price_twd"] >= 75
        and values["quantity_units"] >= 10_000
        and values["delivery_weeks"] <= 5
        and values["payment_months"] >= 1
        and values["contract_years"] >= 1
    )


def ideal_offer(role: str) -> list[float]:
    return [0.0 if pref == "lower" else 1.0 for pref in ROLE_PREFS[role]]


def concession_direction(role: str, issue_idx: int) -> float:
    return 1.0 if ROLE_PREFS[role][issue_idx] == "lower" else -1.0


def apply_action(role: str, offer: list[float], action: int, round_i: int, safe: bool) -> list[float]:
    next_offer = offer[:]
    if action in {HOLD, STOP}:
        return next_offer

    weights = ROLE_WEIGHTS[role]
    low_priority = sorted(range(len(ISSUES)), key=lambda i: weights[i])
    high_priority = sorted(range(len(ISSUES)), key=lambda i: weights[i], reverse=True)

    if action == SMALL:
        step = 0.026 if safe else 0.038
        affected = low_priority[:1]
    elif action == MEDIUM:
        step = 0.043 if safe else 0.060
        affected = low_priority[:2]
    else:
        step = 0.065 if safe else 0.080
        affected = low_priority[:2]

    for idx in affected:
        next_offer[idx] = clip(next_offer[idx] + concession_direction(role, idx) * step)

    if action == TRADE:
        for idx in high_priority[:2]:
            next_offer[idx] = clip(next_offer[idx] - concession_direction(role, idx) * 0.025)

    return next_offer


def opponent_response(
    role: str,
    agent_offer: list[float],
    opponent_offer: list[float],
    opponent_type: OpponentType,
    round_i: int,
    rng: random.Random,
) -> list[float]:
    opponent_role = "seller" if role == "buyer" else "buyer"
    previous_agent_utility = utility(role, opponent_offer)
    pressure_gap = max(0.0, utility(role, agent_offer) - previous_agent_utility)
    deadline_pressure = round_i / MAX_ROUNDS
    concession = opponent_type.concession_mean + 0.08 * pressure_gap + opponent_type.agreement_bias * deadline_pressure
    new_offer = opponent_offer[:]

    for idx in range(len(ISSUES)):
        issue_noise = rng.gauss(0, opponent_type.volatility)
        issue_weight = ROLE_WEIGHTS[opponent_role][idx]
        issue_concession = concession * (1.15 - issue_weight) + issue_noise
        new_offer[idx] = clip(new_offer[idx] + concession_direction(opponent_role, idx) * issue_concession)

    if rng.random() < opponent_type.agreement_bias + 0.05 * deadline_pressure:
        blend = 0.14 + 0.18 * rng.random()
        new_offer = [clip((1 - blend) * x + blend * y) for x, y in zip(new_offer, agent_offer)]

    observed_gain = max(0.0, utility(role, new_offer) - previous_agent_utility)
    return new_offer, observed_gain


def belief_entropy(belief: list[float]) -> float:
    return -sum(p * math.log(max(p, 1e-12)) for p in belief) / math.log(len(belief))


def update_belief(belief: list[float], observed_gain: float) -> list[float]:
    likelihoods = []
    for opponent_type in OPPONENT_TYPES:
        var = max(opponent_type.concession_sd**2, 1e-4)
        like = math.exp(-((observed_gain - opponent_type.concession_mean) ** 2) / (2 * var)) / math.sqrt(2 * math.pi * var)
        likelihoods.append(max(like, 1e-8))
    posterior = [p * l for p, l in zip(belief, likelihoods)]
    total = sum(posterior)
    return [p / total for p in posterior]


def expected_future_gain(belief: list[float], round_i: int) -> float:
    remaining = max(MAX_ROUNDS - round_i, 0)
    expected_concession = sum(p * t.concession_mean for p, t in zip(belief, OPPONENT_TYPES))
    return min(0.18, expected_concession * math.sqrt(remaining) * 0.75)


def opponent_risk(belief: list[float]) -> float:
    return sum(p * (t.pressure + t.volatility) / 1.05 for p, t in zip(belief, OPPONENT_TYPES))


def baseline_state(role: str, own_offer: list[float], opponent_offer: list[float], round_i: int) -> tuple[int, int, int]:
    own_u = utility(role, opponent_offer)
    round_bin = min(5, round_i // 2)
    utility_bin = min(9, int(own_u * 10))
    gap = utility(role, own_offer) - own_u
    gap_bin = 0 if gap < 0.08 else 1 if gap < 0.22 else 2
    return round_bin, utility_bin, gap_bin


def bayesian_state(
    role: str,
    own_offer: list[float],
    opponent_offer: list[float],
    round_i: int,
    belief: list[float],
) -> tuple[int, int, int, int, int, int]:
    own_u = utility(role, opponent_offer)
    opponent_role = "seller" if role == "buyer" else "buyer"
    counter_u = utility(opponent_role, opponent_offer)
    round_bin = min(5, round_i // 2)
    own_bin = min(9, int(own_u * 10))
    counter_bin = min(9, int(counter_u * 10))
    type_bin = max(range(len(belief)), key=lambda i: belief[i])
    entropy_bin = 1 if belief_entropy(belief) > 0.62 else 0
    risk_bin = min(2, int(opponent_risk(belief) * 3))
    return round_bin, own_bin, counter_bin, type_bin, entropy_bin, risk_bin


def safe_actions(
    role: str,
    own_offer: list[float],
    opponent_offer: list[float],
    belief: list[float],
    round_i: int,
    cvar_floor: float | None = None,
) -> list[int]:
    actions = ACTIONS[:]
    if not mandate_ok(role, opponent_offer):
        actions = [a for a in actions if a != STOP]

    risk = opponent_risk(belief)
    entropy = belief_entropy(belief)
    own_u = utility(role, opponent_offer)
    if risk > 0.72 and entropy > 0.48:
        actions = [a for a in actions if a not in {MEDIUM, TRADE}]
    if cvar_floor is not None and cvar_floor < CVaR_SAFE_FLOOR:
        actions = [a for a in actions if a not in {MEDIUM, TRADE}]
    if own_u < 0.58 and round_i < MAX_ROUNDS - 2:
        actions = [a for a in actions if a != STOP]
    if not actions:
        return [HOLD]
    return actions


def q_select(q: dict, st: tuple, actions: list[int], rng: random.Random, eps: float) -> int:
    if rng.random() < eps:
        return rng.choice(actions)
    return max(actions, key=lambda a: q.get((st, a), 0.0))


def baseline_accept(role: str, incoming: list[float], round_i: int) -> bool:
    if not mandate_ok(role, incoming):
        return False
    threshold = max(0.64, 0.80 - 0.025 * round_i)
    return utility(role, incoming) >= threshold


def bayesian_safe_accept(role: str, incoming: list[float], round_i: int, belief: list[float]) -> bool:
    if not mandate_ok(role, incoming):
        return False
    current = utility(role, incoming)
    if round_i >= MAX_ROUNDS - 1:
        return current >= 0.67
    reservation_threshold = max(0.67, 0.80 - 0.025 * round_i)
    continuation_value = current + expected_future_gain(belief, round_i) - 0.025 * (MAX_ROUNDS - round_i)
    uncertainty_premium = 0.012 if belief_entropy(belief) > 0.58 and round_i < MAX_ROUNDS - 3 else 0.0
    return current >= reservation_threshold + uncertainty_premium and current >= continuation_value - 0.035


def empirical_cvar(samples: list[float], alpha: float = CVaR_ALPHA) -> float:
    if not samples:
        return CVaR_SAFE_FLOOR
    ordered = sorted(samples)
    n_tail = max(1, math.ceil(len(ordered) * alpha))
    return mean(ordered[:n_tail])


def cvar_safe_accept(role: str, incoming: list[float], round_i: int, belief: list[float], cvar_floor: float) -> bool:
    if not mandate_ok(role, incoming):
        return False
    current = utility(role, incoming)
    tail_deficit = max(CVaR_SAFE_FLOOR - cvar_floor, 0.0)
    if round_i >= MAX_ROUNDS - 1:
        return current >= max(0.69, CVaR_SAFE_FLOOR + 0.25 * tail_deficit)
    reservation_threshold = max(0.69, 0.82 - 0.020 * round_i + 0.35 * tail_deficit)
    continuation_value = current + expected_future_gain(belief, round_i) - 0.022 * (MAX_ROUNDS - round_i)
    uncertainty_premium = 0.020 * belief_entropy(belief) + 0.025 * opponent_risk(belief)
    return current >= reservation_threshold + uncertainty_premium and current >= continuation_value - 0.020


def run_episode(
    role: str,
    q: dict | None,
    algorithm: str,
    rng: random.Random,
    train: bool,
    cvar_floor: float = CVaR_SAFE_FLOOR,
) -> dict:
    opponent_type = rng.choice(OPPONENT_TYPES)
    opponent_role = "seller" if role == "buyer" else "buyer"
    own_offer = ideal_offer(role)
    opponent_offer = ideal_offer(opponent_role)
    belief = [1 / len(OPPONENT_TYPES)] * len(OPPONENT_TYPES)
    alpha = 0.16 if algorithm == "bayesian_safe_q" else 0.18
    gamma = 0.94 if algorithm == "bayesian_safe_q" else 0.92
    eps = 0.18 if train else 0.0
    action_names = []

    for round_i in range(MAX_ROUNDS):
        if algorithm == "pomdp_cvar_safe_q":
            accepted = cvar_safe_accept(role, opponent_offer, round_i, belief, cvar_floor)
            st = bayesian_state(role, own_offer, opponent_offer, round_i, belief)
            available = safe_actions(role, own_offer, opponent_offer, belief, round_i, cvar_floor)
        elif algorithm == "bayesian_safe_q":
            accepted = bayesian_safe_accept(role, opponent_offer, round_i, belief)
            st = bayesian_state(role, own_offer, opponent_offer, round_i, belief)
            available = safe_actions(role, own_offer, opponent_offer, belief, round_i)
        else:
            accepted = baseline_accept(role, opponent_offer, round_i)
            st = baseline_state(role, own_offer, opponent_offer, round_i)
            available = ACTIONS[:]

        if accepted:
            own_u = utility(role, opponent_offer)
            opp_u = utility(opponent_role, opponent_offer)
            return {
                "algorithm": algorithm,
                "role": role,
                "opponent_type": opponent_type.name,
                "outcome": "agreement",
                "rounds": round_i + 1,
                "own_utility": own_u,
                "counterpart_utility": opp_u,
                "joint_utility": own_u + opp_u,
                "target_met": own_u >= TARGET_UTILITY,
                "mandate_violation": not mandate_ok(role, opponent_offer),
                "belief_entropy": belief_entropy(belief),
                "actions": ";".join(action_names + ["accept"]),
                **concrete_offer(opponent_offer),
            }

        action = q_select(q or {}, st, available, rng, eps)
        action_names.append(ACTION_NAMES[action])
        if action == STOP:
            if train and q is not None:
                q[(st, action)] = q.get((st, action), 0.0) + alpha * (DISAGREEMENT_UTILITY - q.get((st, action), 0.0))
            return {
                "algorithm": algorithm,
                "role": role,
                "opponent_type": opponent_type.name,
                "outcome": "stop",
                "rounds": round_i + 1,
                "own_utility": DISAGREEMENT_UTILITY,
                "counterpart_utility": DISAGREEMENT_UTILITY,
                "joint_utility": 2 * DISAGREEMENT_UTILITY,
                "target_met": False,
                "mandate_violation": False,
                "belief_entropy": belief_entropy(belief),
                "actions": ";".join(action_names),
                **concrete_offer(opponent_offer),
            }

        next_own_offer = apply_action(role, own_offer, action, round_i, safe=(algorithm == "bayesian_safe_q"))
        next_opponent_offer, observed_gain = opponent_response(role, next_own_offer, opponent_offer, opponent_type, round_i, rng)
        next_belief = update_belief(belief, observed_gain) if algorithm in {"bayesian_safe_q", "pomdp_cvar_safe_q"} else belief

        own_u_now = utility(role, next_opponent_offer)
        opp_u_now = utility(opponent_role, next_opponent_offer)
        concession_cost = max(0.0, utility(role, own_offer) - utility(role, next_own_offer))
        target_gap = max(TARGET_UTILITY - own_u_now, 0.0)
        unsafe = not mandate_ok(role, next_opponent_offer)
        instability = opponent_risk(next_belief) * belief_entropy(next_belief) if algorithm in {"bayesian_safe_q", "pomdp_cvar_safe_q"} else 0.0

        if algorithm == "pomdp_cvar_safe_q":
            cvar_deficit = max(CVaR_SAFE_FLOOR - cvar_floor, 0.0)
            tail_loss = max(CVaR_SAFE_FLOOR - own_u_now, 0.0)
            reward = (
                1.40 * own_u_now
                + 0.08 * (own_u_now + opp_u_now)
                + 0.14 * max(own_u_now - DISAGREEMENT_UTILITY, 0) * max(opp_u_now - DISAGREEMENT_UTILITY, 0)
                - 1.05 * concession_cost
                - 0.60 * target_gap
                - 0.45 * tail_loss
                - 0.35 * cvar_deficit
                - 0.24 * instability
                - 6.0 * unsafe
                - 0.014 * round_i
            )
            next_st = bayesian_state(role, next_own_offer, next_opponent_offer, round_i + 1, next_belief)
            next_actions = safe_actions(role, next_own_offer, next_opponent_offer, next_belief, round_i + 1, cvar_floor)
        elif algorithm == "bayesian_safe_q":
            reward = (
                1.30 * own_u_now
                + 0.10 * (own_u_now + opp_u_now)
                + 0.18 * max(own_u_now - DISAGREEMENT_UTILITY, 0) * max(opp_u_now - DISAGREEMENT_UTILITY, 0)
                - 0.95 * concession_cost
                - 0.55 * target_gap
                - 0.20 * instability
                - 5.0 * unsafe
                - 0.015 * round_i
            )
            next_st = bayesian_state(role, next_own_offer, next_opponent_offer, round_i + 1, next_belief)
            next_actions = safe_actions(role, next_own_offer, next_opponent_offer, next_belief, round_i + 1)
        else:
            reward = (
                1.05 * own_u_now
                + 0.12 * (own_u_now + opp_u_now)
                - 0.65 * concession_cost
                - 0.35 * target_gap
                - 3.0 * unsafe
                - 0.018 * round_i
            )
            next_st = baseline_state(role, next_own_offer, next_opponent_offer, round_i + 1)
            next_actions = ACTIONS[:]

        if train and q is not None:
            old = q.get((st, action), 0.0)
            future = max(q.get((next_st, a), 0.0) for a in next_actions)
            q[(st, action)] = old + alpha * (reward + gamma * future - old)

        own_offer = next_own_offer
        opponent_offer = next_opponent_offer
        belief = next_belief

    own_u = utility(role, opponent_offer)
    opp_u = utility(opponent_role, opponent_offer)
    return {
        "algorithm": algorithm,
        "role": role,
        "opponent_type": opponent_type.name,
        "outcome": "deadline",
        "rounds": MAX_ROUNDS,
        "own_utility": own_u,
        "counterpart_utility": opp_u,
        "joint_utility": own_u + opp_u,
        "target_met": False,
        "mandate_violation": not mandate_ok(role, opponent_offer),
        "belief_entropy": belief_entropy(belief),
        "actions": ";".join(action_names),
        **concrete_offer(opponent_offer),
    }


def train(role: str, algorithm: str) -> tuple[dict, float]:
    rng = random.Random(SEED + hash((role, algorithm)) % 10000)
    q = {}
    terminal_utilities: list[float] = []
    cvar_floor = CVaR_SAFE_FLOOR
    for _ in range(TRAIN_EPISODES):
        record = run_episode(role, q, algorithm, rng, train=True, cvar_floor=cvar_floor)
        terminal_utilities.append(record["own_utility"])
        if algorithm == "pomdp_cvar_safe_q" and len(terminal_utilities) >= 200 and len(terminal_utilities) % 200 == 0:
            cvar_floor = empirical_cvar(terminal_utilities[-2000:])
    return q, cvar_floor


def evaluate(role: str, algorithm: str, q: dict, cvar_floor: float) -> list[dict]:
    rng = random.Random(SEED + 100000 + hash((role, algorithm)) % 10000)
    return [run_episode(role, q, algorithm, rng, train=False, cvar_floor=cvar_floor) for _ in range(EVAL_EPISODES)]


def summarize(records: list[dict]) -> dict:
    utilities = [r["own_utility"] for r in records]
    agreements = [r for r in records if r["outcome"] == "agreement"]
    successful_agreements = [r for r in agreements if r["target_met"]]
    return {
        "algorithm": records[0]["algorithm"],
        "role": records[0]["role"],
        "n": len(records),
        "agreement_rate": len(agreements) / len(records),
        "target_met_rate": sum(r["target_met"] for r in records) / len(records),
        "successful_agreement_rate": len(successful_agreements) / len(records),
        "target_given_agreement": len(successful_agreements) / len(agreements) if agreements else 0.0,
        "mandate_violation_rate": sum(r["mandate_violation"] for r in records) / len(records),
        "mean_own_utility": mean(utilities),
        "utility_sd": pstdev(utilities),
        "mean_counterpart_utility": mean(r["counterpart_utility"] for r in records),
        "mean_joint_utility": mean(r["joint_utility"] for r in records),
        "mean_rounds": mean(r["rounds"] for r in records),
        "stop_rate": sum(r["outcome"] == "stop" for r in records) / len(records),
        "deadline_rate": sum(r["outcome"] == "deadline" for r in records) / len(records),
        "mean_belief_entropy": mean(r["belief_entropy"] for r in records),
        "empirical_cvar_10": empirical_cvar(utilities),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    all_records = []
    summary_rows = []
    q_meta = {}
    for algorithm in ["principal_q", "bayesian_safe_q", "pomdp_cvar_safe_q"]:
        for role in ["buyer", "seller"]:
            print(f"Training {algorithm} as {role} for {TRAIN_EPISODES:,} episodes...", flush=True)
            q, cvar_floor = train(role, algorithm)
            print(f"Evaluating {algorithm} as {role} for {EVAL_EPISODES:,} episodes...", flush=True)
            records = evaluate(role, algorithm, q, cvar_floor)
            all_records.extend(records)
            summary_rows.append(summarize(records))
            q_meta[f"{algorithm}_{role}"] = {
                "q_entries": len(q),
                "q_states": len({st for st, _a in q}),
                "learned_cvar_floor": cvar_floor,
            }

    write_csv(OUT_DIR / "bayesian_safe_rl_episode_results.csv", all_records)
    write_csv(OUT_DIR / "bayesian_safe_rl_summary.csv", summary_rows)
    with (OUT_DIR / "bayesian_safe_rl_q_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(q_meta, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary_rows, ensure_ascii=False, indent=2))
    print(f"Wrote {OUT_DIR / 'bayesian_safe_rl_summary.csv'}")


if __name__ == "__main__":
    main()
