from __future__ import annotations

import csv
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev


SEED = 20260513
TRAIN_EPISODES = int(os.environ.get("NEGOTIATION_TRAIN_EPISODES", "30000"))
EVAL_EPISODES = int(os.environ.get("NEGOTIATION_EVAL_EPISODES", "1000"))
MAX_ROUNDS = 10

OUT_DIR = Path("_negotiation_simulation")
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

DISAGREEMENT_UTILITY = 0.20
TARGET_UTILITY = 0.70
INITIAL_ACCEPT_UTILITY = 0.82
MIN_ACCEPT_UTILITY = 0.66

HOLD = 0
SMALL = 1
MEDIUM = 2
INTEGRATIVE = 3
STOP = 4
ACTIONS = [HOLD, SMALL, MEDIUM, INTEGRATIVE, STOP]
ACTION_NAMES = {
    HOLD: "hold",
    SMALL: "small_concession",
    MEDIUM: "medium_concession",
    INTEGRATIVE: "integrative_trade",
    STOP: "stop",
}


@dataclass(frozen=True)
class ChatGPTType:
    name: str
    agreement_bias: float
    pressure: float
    concession_noise: float
    concession_base: float


CHATGPT_TYPES = [
    ChatGPTType("agreeable", 0.20, 0.35, 0.06, 0.075),
    ChatGPTType("balanced", 0.10, 0.55, 0.05, 0.045),
    ChatGPTType("pressure_sensitive", 0.26, 0.80, 0.08, 0.090),
    ChatGPTType("inconsistent", 0.12, 0.65, 0.13, 0.055),
]


def clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def score_issue(z: float, pref: str) -> float:
    return z if pref == "higher" else 1.0 - z


def utility(role: str, offer: list[float]) -> float:
    weights = ROLE_WEIGHTS[role]
    prefs = ROLE_PREFS[role]
    return sum(w * score_issue(z, pref) for w, z, pref in zip(weights, offer, prefs))


def concrete_offer(offer: list[float]) -> dict[str, float]:
    return {
        issue.name: issue.low + z * (issue.high - issue.low)
        for issue, z in zip(ISSUES, offer)
    }


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
    # Direction that concedes value to the counterpart.
    return 1.0 if ROLE_PREFS[role][issue_idx] == "lower" else -1.0


def apply_action(role: str, offer: list[float], action: int, round_i: int, rng: random.Random) -> list[float]:
    next_offer = offer[:]
    if action == HOLD:
        return next_offer
    if action == STOP:
        return next_offer

    weights = ROLE_WEIGHTS[role]
    if action == SMALL:
        issue_order = sorted(range(len(ISSUES)), key=lambda i: weights[i])
        step = 0.035
        chosen = issue_order[0]
        next_offer[chosen] = clip(next_offer[chosen] + concession_direction(role, chosen) * step)
    elif action == MEDIUM:
        issue_order = sorted(range(len(ISSUES)), key=lambda i: weights[i])
        for chosen in issue_order[:2]:
            next_offer[chosen] = clip(next_offer[chosen] + concession_direction(role, chosen) * 0.055)
    elif action == INTEGRATIVE:
        # Trade lower-weight concessions for holding high-weight issues.
        low_priority = sorted(range(len(ISSUES)), key=lambda i: weights[i])[:2]
        high_priority = sorted(range(len(ISSUES)), key=lambda i: weights[i], reverse=True)[:2]
        for chosen in low_priority:
            next_offer[chosen] = clip(next_offer[chosen] + concession_direction(role, chosen) * 0.075)
        for chosen in high_priority:
            next_offer[chosen] = clip(next_offer[chosen] - concession_direction(role, chosen) * 0.020)

    # Small stochastic implementation variance.
    jitter_idx = rng.randrange(len(ISSUES))
    next_offer[jitter_idx] = clip(next_offer[jitter_idx] + rng.gauss(0, 0.006))
    return next_offer


def chatgpt_counter(role: str, agent_offer: list[float], opponent_offer: list[float], ctype: ChatGPTType, round_i: int, rng: random.Random) -> list[float]:
    opponent_role = "seller" if role == "buyer" else "buyer"
    new_offer = opponent_offer[:]
    agent_pressure = max(0.0, utility(role, agent_offer) - utility(role, opponent_offer))
    deadline_pressure = round_i / MAX_ROUNDS
    concession = ctype.concession_base + ctype.agreement_bias * deadline_pressure + 0.08 * agent_pressure

    for idx in range(len(ISSUES)):
        direction = concession_direction(opponent_role, idx)
        issue_noise = rng.gauss(0, ctype.concession_noise)
        issue_weight = ROLE_WEIGHTS[opponent_role][idx]
        issue_concession = concession * (1.20 - issue_weight) + issue_noise
        new_offer[idx] = clip(new_offer[idx] + direction * issue_concession)

    # Simulated ChatGPT sometimes prioritizes agreement language over mandate fidelity.
    if rng.random() < (ctype.agreement_bias + 0.08 * deadline_pressure):
        blend = 0.18 + 0.20 * rng.random()
        new_offer = [clip((1 - blend) * x + blend * y) for x, y in zip(new_offer, agent_offer)]

    return new_offer


def state(role: str, own_offer: list[float], opponent_offer: list[float], round_i: int) -> tuple[int, int, int, int]:
    own_u = utility(role, opponent_offer)
    counter_u = utility("seller" if role == "buyer" else "buyer", opponent_offer)
    own_bin = min(9, int(own_u * 10))
    counter_bin = min(9, int(counter_u * 10))
    round_bin = min(5, round_i // 2)
    gap = utility(role, own_offer) - own_u
    gap_bin = 0 if gap < 0.08 else 1 if gap < 0.22 else 2
    return round_bin, own_bin, counter_bin, gap_bin


def state_key(st: tuple[int, int, int, int]) -> str:
    return "|".join(str(x) for x in st)


def policy_from_q(q: dict[tuple[tuple[int, int, int, int], int], float]) -> dict[str, str]:
    actions_by_state: dict[tuple[int, int, int, int], set[int]] = {}
    for st, action in q:
        actions_by_state.setdefault(st, set()).add(action)
    return {
        state_key(st): ACTION_NAMES[max(actions, key=lambda action: q.get((st, action), 0.0))]
        for st, actions in actions_by_state.items()
    }


def admissible_actions(role: str, opponent_offer: list[float]) -> list[int]:
    actions = ACTIONS[:]
    # The accept decision is implicit: accept if the incoming offer clears the learned threshold.
    if not mandate_ok(role, opponent_offer):
        return [a for a in actions if a != STOP]
    return actions


def q_select(q: dict[tuple[tuple[int, int, int, int], int], float], st: tuple[int, int, int, int], actions: list[int], rng: random.Random, eps: float) -> int:
    if rng.random() < eps:
        return rng.choice(actions)
    return max(actions, key=lambda a: q.get((st, a), 0.0))


def should_accept(role: str, incoming_offer: list[float], round_i: int, learned_margin: float = 0.0) -> bool:
    if not mandate_ok(role, incoming_offer):
        return False
    threshold = INITIAL_ACCEPT_UTILITY - 0.020 * round_i + learned_margin
    threshold = max(MIN_ACCEPT_UTILITY, threshold)
    return utility(role, incoming_offer) >= threshold


def episode(role: str, q: dict | None, rng: random.Random, train: bool = False) -> dict:
    ctype = rng.choice(CHATGPT_TYPES)
    own_offer = ideal_offer(role)
    opponent_role = "seller" if role == "buyer" else "buyer"
    opponent_offer = ideal_offer(opponent_role)
    alpha = 0.18
    gamma = 0.92
    eps = 0.16 if train else 0.0
    total_reward = 0.0
    actions_taken: list[str] = []

    for round_i in range(MAX_ROUNDS):
        if should_accept(role, opponent_offer, round_i):
            own_u = utility(role, opponent_offer)
            opp_u = utility(opponent_role, opponent_offer)
            violation = not mandate_ok(role, opponent_offer)
            target_bonus = 0.35 if own_u >= TARGET_UTILITY else -0.45 * (TARGET_UTILITY - own_u)
            reward = 1.65 * own_u + 0.14 * (own_u + opp_u) + target_bonus - 5.0 * violation - 0.012 * round_i
            total_reward += reward
            return {
                "role": role,
                "chatgpt_type": ctype.name,
                "outcome": "agreement",
                "rounds": round_i + 1,
                "own_utility": own_u,
                "counterpart_utility": opp_u,
                "joint_utility": own_u + opp_u,
                "mandate_violation": violation,
                "target_met": own_u >= TARGET_UTILITY and not violation,
                "actions": ";".join(actions_taken + ["accept"]),
                **concrete_offer(opponent_offer),
            }

        st = state(role, own_offer, opponent_offer, round_i)
        actions = admissible_actions(role, opponent_offer)
        action = q_select(q or {}, st, actions, rng, eps)
        actions_taken.append(ACTION_NAMES[action])

        if action == STOP:
            own_u = DISAGREEMENT_UTILITY
            opp_u = DISAGREEMENT_UTILITY
            reward = own_u - 0.03 * round_i
            if train and q is not None:
                q[(st, action)] = q.get((st, action), 0.0) + alpha * (reward - q.get((st, action), 0.0))
            return {
                "role": role,
                "chatgpt_type": ctype.name,
                "outcome": "stop",
                "rounds": round_i + 1,
                "own_utility": own_u,
                "counterpart_utility": opp_u,
                "joint_utility": own_u + opp_u,
                "mandate_violation": False,
                "target_met": False,
                "actions": ";".join(actions_taken),
                **concrete_offer(opponent_offer),
            }

        next_own_offer = apply_action(role, own_offer, action, round_i, rng)
        next_opponent_offer = chatgpt_counter(role, next_own_offer, opponent_offer, ctype, round_i, rng)
        next_st = state(role, next_own_offer, next_opponent_offer, round_i + 1)

        own_u_now = utility(role, next_opponent_offer)
        opp_u_now = utility(opponent_role, next_opponent_offer)
        concession_cost = max(0.0, utility(role, own_offer) - utility(role, next_own_offer))
        unsafe = not mandate_ok(role, next_opponent_offer)
        target_gap = max(TARGET_UTILITY - own_u_now, 0.0)
        reward = (
            1.10 * own_u_now
            + 0.10 * (own_u_now + opp_u_now)
            + 0.16 * max(own_u_now - DISAGREEMENT_UTILITY, 0) * max(opp_u_now - DISAGREEMENT_UTILITY, 0)
            - 0.85 * concession_cost
            - 0.45 * target_gap
            - 0.018 * round_i
            - 4.0 * unsafe
        )

        if train and q is not None:
            old = q.get((st, action), 0.0)
            future = max(q.get((next_st, a), 0.0) for a in admissible_actions(role, next_opponent_offer))
            q[(st, action)] = old + alpha * (reward + gamma * future - old)

        total_reward += reward
        own_offer = next_own_offer
        opponent_offer = next_opponent_offer

    own_u = utility(role, opponent_offer)
    opp_u = utility(opponent_role, opponent_offer)
    violation = not mandate_ok(role, opponent_offer)
    return {
        "role": role,
        "chatgpt_type": ctype.name,
        "outcome": "deadline",
        "rounds": MAX_ROUNDS,
        "own_utility": own_u,
        "counterpart_utility": opp_u,
        "joint_utility": own_u + opp_u,
        "mandate_violation": violation,
        "target_met": own_u >= TARGET_UTILITY and not violation,
        "actions": ";".join(actions_taken),
        **concrete_offer(opponent_offer),
    }


def train(role: str) -> dict:
    rng = random.Random(SEED + (1 if role == "buyer" else 2))
    q: dict[tuple[tuple[int, int, int, int], int], float] = {}
    for _ in range(TRAIN_EPISODES):
        episode(role, q, rng, train=True)
    return q


def evaluate(role: str, q: dict) -> list[dict]:
    rng = random.Random(SEED + (101 if role == "buyer" else 202))
    return [episode(role, q, rng, train=False) for _ in range(EVAL_EPISODES)]


def summarize(records: list[dict]) -> dict[str, str]:
    utilities = [r["own_utility"] for r in records]
    agreements = [r for r in records if r["outcome"] == "agreement"]
    return {
        "role": records[0]["role"],
        "n": str(len(records)),
        "agreement_rate": f"{len(agreements) / len(records):.1%}",
        "target_met_rate": f"{sum(r['target_met'] for r in records) / len(records):.1%}",
        "mandate_violation_rate": f"{sum(r['mandate_violation'] for r in records) / len(records):.1%}",
        "mean_own_utility": f"{mean(utilities):.3f}",
        "utility_sd": f"{pstdev(utilities):.3f}",
        "mean_counterpart_utility": f"{mean(r['counterpart_utility'] for r in records):.3f}",
        "mean_joint_utility": f"{mean(r['joint_utility'] for r in records):.3f}",
        "mean_rounds": f"{mean(r['rounds'] for r in records):.2f}",
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    all_rows: list[dict] = []
    summary_rows: list[dict[str, str]] = []
    policies = {
        "metadata": {
            "seed": SEED,
            "train_episodes_per_role": TRAIN_EPISODES,
            "target_utility": TARGET_UTILITY,
            "initial_accept_utility": INITIAL_ACCEPT_UTILITY,
            "min_accept_utility": MIN_ACCEPT_UTILITY,
            "actions": ACTION_NAMES,
            "state_key": "round_bin|own_utility_bin|counterpart_utility_bin|gap_bin",
        },
        "roles": {},
    }
    for role in ["buyer", "seller"]:
        q = train(role)
        records = evaluate(role, q)
        all_rows.extend(records)
        summary_rows.append(summarize(records))
        policies["roles"][role] = {
            "q_states": len({state_key(st) for st, _action in q.keys()}),
            "policy": policy_from_q(q),
        }

    write_csv(OUT_DIR / "five_issue_simulation_episodes.csv", all_rows)
    write_csv(OUT_DIR / "five_issue_simulation_summary.csv", summary_rows)
    with (OUT_DIR / "principal_first_policy.json").open("w", encoding="utf-8") as f:
        json.dump(policies, f, ensure_ascii=False, indent=2)

    print("Five-issue constrained MDP-RL negotiation simulation")
    print(f"Training episodes per role: {TRAIN_EPISODES:,}")
    print(f"Evaluation episodes per role: {EVAL_EPISODES:,}")
    print()
    for row in summary_rows:
        print(row)
    print()
    print(f"Wrote {OUT_DIR / 'five_issue_simulation_episodes.csv'}")
    print(f"Wrote {OUT_DIR / 'five_issue_simulation_summary.csv'}")
    print(f"Wrote {OUT_DIR / 'principal_first_policy.json'}")


if __name__ == "__main__":
    main()
