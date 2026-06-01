from __future__ import annotations

import csv
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev


SEED = 20260513
TOTAL_TRIALS = int(os.environ.get("NEGOTIATION_TRIALS", "10"))
MAX_ROUNDS = int(os.environ.get("NEGOTIATION_MAX_ROUNDS", "6"))
MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.5")
API_URL = "https://api.openai.com/v1/responses"
OUT_DIR = Path("_negotiation_api_experiment")
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

TARGET_UTILITY = 0.70
POLICY_PATH = Path(os.environ.get("NEGOTIATION_POLICY_PATH", "_negotiation_simulation/principal_first_policy.json"))
INITIAL_ACCEPT_UTILITY = 0.82
MIN_ACCEPT_UTILITY = 0.66


def clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def norm_to_value(issue: Issue, z: float) -> float:
    return issue.low + z * (issue.high - issue.low)


def value_to_norm(issue: Issue, value: float) -> float:
    return clip((value - issue.low) / (issue.high - issue.low))


def score_issue(z: float, pref: str) -> float:
    return z if pref == "higher" else 1.0 - z


def utility(role: str, offer: dict[str, float]) -> float:
    total = 0.0
    for issue, weight, pref in zip(ISSUES, ROLE_WEIGHTS[role], ROLE_PREFS[role]):
        total += weight * score_issue(value_to_norm(issue, float(offer[issue.name])), pref)
    return total


def mandate_ok(role: str, offer: dict[str, float]) -> bool:
    if role == "buyer":
        return (
            offer["unit_price_twd"] <= 500
            and offer["quantity_units"] <= 50_000
            and offer["delivery_weeks"] <= 5
            and offer["payment_months"] >= 1
            and offer["contract_years"] >= 1
        )
    return (
        offer["unit_price_twd"] >= 75
        and offer["quantity_units"] >= 10_000
        and offer["delivery_weeks"] <= 5
        and offer["payment_months"] >= 1
        and offer["contract_years"] >= 1
    )


def ideal_offer(role: str) -> dict[str, float]:
    out = {}
    for issue, pref in zip(ISSUES, ROLE_PREFS[role]):
        z = 0.0 if pref == "lower" else 1.0
        out[issue.name] = round(norm_to_value(issue, z), 3)
    return out


def bounded_offer(offer: dict[str, float]) -> dict[str, float]:
    out = {}
    for issue in ISSUES:
        value = float(offer.get(issue.name, issue.low))
        if issue.name == "quantity_units":
            value = round(value)
        out[issue.name] = round(max(issue.low, min(issue.high, value)), 3)
    return out


def concession_direction(role: str, issue_name: str) -> int:
    idx = [i.name for i in ISSUES].index(issue_name)
    return 1 if ROLE_PREFS[role][idx] == "lower" else -1


def agent_counteroffer(role: str, current_offer: dict[str, float], incoming: dict[str, float], round_i: int) -> dict[str, float]:
    return agent_counteroffer_by_action(role, current_offer, "small_concession" if round_i else "hold", round_i)


def agent_counteroffer_by_action(role: str, current_offer: dict[str, float], action: str, round_i: int) -> dict[str, float]:
    if round_i == 0 or action == "hold":
        return current_offer
    if action == "stop":
        return current_offer

    next_offer = dict(current_offer)
    weights = {issue.name: weight for issue, weight in zip(ISSUES, ROLE_WEIGHTS[role])}
    low_priority = sorted(weights, key=weights.get)[:2]
    step_by_action = {
        "small_concession": 0.035,
        "medium_concession": 0.055,
        "integrative_trade": 0.075,
    }
    step = step_by_action.get(action, 0.035)

    affected = low_priority[:1] if action == "small_concession" else low_priority
    for issue_name in affected:
        issue = next(i for i in ISSUES if i.name == issue_name)
        z = value_to_norm(issue, next_offer[issue_name])
        z = clip(z + concession_direction(role, issue_name) * step)
        next_offer[issue_name] = round(norm_to_value(issue, z), 3)

    if action == "integrative_trade":
        high_priority = sorted(weights, key=weights.get, reverse=True)[:2]
        for issue_name in high_priority:
            issue = next(i for i in ISSUES if i.name == issue_name)
            z = value_to_norm(issue, next_offer[issue_name])
            z = clip(z - concession_direction(role, issue_name) * 0.020)
            next_offer[issue_name] = round(norm_to_value(issue, z), 3)
    elif action == "medium_concession":
        medium_issue = sorted(weights, key=weights.get)[2]
        issue = next(i for i in ISSUES if i.name == medium_issue)
        z = value_to_norm(issue, next_offer[medium_issue])
        z = clip(z + concession_direction(role, medium_issue) * 0.025)
        next_offer[medium_issue] = round(norm_to_value(issue, z), 3)
    return bounded_offer(next_offer)


def state_key(role: str, own_offer: dict[str, float], opponent_offer: dict[str, float], round_i: int) -> str:
    own_u = utility(role, opponent_offer)
    opponent_role = "seller" if role == "buyer" else "buyer"
    counter_u = utility(opponent_role, opponent_offer)
    own_bin = min(9, int(own_u * 10))
    counter_bin = min(9, int(counter_u * 10))
    round_bin = min(5, round_i // 2)
    gap = utility(role, own_offer) - own_u
    gap_bin = 0 if gap < 0.08 else 1 if gap < 0.22 else 2
    return f"{round_bin}|{own_bin}|{counter_bin}|{gap_bin}"


def load_policy() -> dict:
    if not POLICY_PATH.exists():
        return {}
    with POLICY_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def policy_action(policy: dict, role: str, own_offer: dict[str, float], incoming: dict[str, float], round_i: int) -> str:
    role_policy = policy.get("roles", {}).get(role, {}).get("policy", {})
    key = state_key(role, own_offer, incoming, round_i)
    if key in role_policy:
        return role_policy[key]
    if utility(role, incoming) < MIN_ACCEPT_UTILITY and round_i >= MAX_ROUNDS - 1:
        return "stop"
    return "small_concession"


def should_accept(role: str, incoming_offer: dict[str, float], round_i: int) -> bool:
    if not mandate_ok(role, incoming_offer):
        return False
    threshold = max(MIN_ACCEPT_UTILITY, INITIAL_ACCEPT_UTILITY - 0.020 * round_i)
    return utility(role, incoming_offer) >= threshold


def offer_text(offer: dict[str, float]) -> str:
    return (
        f"unit price {offer['unit_price_twd']:.2f} TWD, "
        f"quantity {offer['quantity_units']:.0f} units, "
        f"delivery {offer['delivery_weeks']:.2f} weeks, "
        f"payment {offer['payment_months']:.2f} months, "
        f"contract {offer['contract_years']:.2f} years"
    )


def api_call(prompt: str, history: list[dict[str, str]]) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    payload = {
        "model": MODEL,
        "input": [{"role": "system", "content": prompt}] + history,
        "temperature": 0.7,
        "max_output_tokens": 450,
        "store": False,
    }
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if "output_text" in data:
        return data["output_text"]
    texts: list[str] = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                texts.append(content.get("text", ""))
    return "\n".join(texts).strip()


def parse_json_offer(text: str) -> dict:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError(f"No JSON object found in model reply: {text[:200]}")
    data = json.loads(match.group(0))
    offer = bounded_offer(data)
    status = str(data.get("agreement_status", "counteroffer")).lower()
    message = str(data.get("message", ""))
    return {"offer": offer, "agreement_status": status, "message": message}


def opponent_prompt(opponent_role: str) -> str:
    if opponent_role == "seller":
        role_limits = (
            "You are the seller. Prefer high unit price, high quantity, longer delivery, "
            "shorter payment, and longer contract. Your hard limits are unit_price_twd >= 75, "
            "quantity_units >= 10000, delivery_weeks <= 5, payment_months >= 1, contract_years >= 1."
        )
    else:
        role_limits = (
            "You are the buyer. Prefer low unit price, low quantity, shorter delivery, "
            "longer payment, and shorter contract. Your hard limits are unit_price_twd <= 500, "
            "quantity_units <= 50000, delivery_weeks <= 5, payment_months >= 1, contract_years >= 1."
        )
    return (
        role_limits
        + "\nNegotiate a five-issue procurement contract. You are cooperative and want agreement, "
        "but protect your side's interests. Reply ONLY as valid JSON with exactly these keys: "
        "unit_price_twd, quantity_units, delivery_weeks, payment_months, contract_years, "
        "agreement_status, message. agreement_status must be counteroffer, accept, or reject."
    )


def run_trial(trial_id: int, agent_role: str, rng: random.Random, policy: dict) -> tuple[list[dict], dict]:
    opponent_role = "seller" if agent_role == "buyer" else "buyer"
    prompt = opponent_prompt(opponent_role)
    agent_offer = ideal_offer(agent_role)
    incoming_offer = ideal_offer(opponent_role)
    history: list[dict[str, str]] = []
    rows: list[dict] = []

    for round_i in range(1, MAX_ROUNDS + 1):
        if should_accept(agent_role, incoming_offer, round_i):
            final = {
                "trial_id": trial_id,
                "agent_role": agent_role,
                "opponent_role": opponent_role,
                "outcome": "agent_accept",
                "rounds": round_i,
                "own_utility": utility(agent_role, incoming_offer),
                "counterpart_utility": utility(opponent_role, incoming_offer),
                "joint_utility": utility(agent_role, incoming_offer) + utility(opponent_role, incoming_offer),
                "mandate_violation": not mandate_ok(agent_role, incoming_offer),
                "target_met": utility(agent_role, incoming_offer) >= TARGET_UTILITY and mandate_ok(agent_role, incoming_offer),
                **incoming_offer,
            }
            return rows, final

        action = policy_action(policy, agent_role, agent_offer, incoming_offer, round_i - 1)
        if action == "stop":
            final = {
                "trial_id": trial_id,
                "agent_role": agent_role,
                "opponent_role": opponent_role,
                "outcome": "agent_stop",
                "rounds": round_i,
                "own_utility": 0.20,
                "counterpart_utility": 0.20,
                "joint_utility": 0.40,
                "mandate_violation": False,
                "target_met": False,
                **incoming_offer,
            }
            return rows, final

        agent_offer = agent_counteroffer_by_action(agent_role, agent_offer, action, round_i - 1)
        user_message = (
            f"Our counteroffer is: {offer_text(agent_offer)}. "
            "Please respond with either accept, reject, or a complete counteroffer in the required JSON format."
        )
        history.append({"role": "user", "content": user_message})
        raw = api_call(prompt, history)
        history.append({"role": "assistant", "content": raw})
        parsed = parse_json_offer(raw)
        incoming_offer = parsed["offer"]

        row = {
            "trial_id": trial_id,
            "agent_role": agent_role,
            "round": round_i,
            "policy_action": action,
            "agent_offer": offer_text(agent_offer),
            "chatgpt_status": parsed["agreement_status"],
            "chatgpt_message": parsed["message"],
            "raw_reply": raw,
            "own_utility": utility(agent_role, incoming_offer),
            "counterpart_utility": utility(opponent_role, incoming_offer),
            "mandate_violation": not mandate_ok(agent_role, incoming_offer),
            **incoming_offer,
        }
        rows.append(row)

        if parsed["agreement_status"] == "accept" and mandate_ok(agent_role, agent_offer):
            final_offer = agent_offer
            final = {
                "trial_id": trial_id,
                "agent_role": agent_role,
                "opponent_role": opponent_role,
                "outcome": "chatgpt_accept",
                "rounds": round_i,
                "own_utility": utility(agent_role, final_offer),
                "counterpart_utility": utility(opponent_role, final_offer),
                "joint_utility": utility(agent_role, final_offer) + utility(opponent_role, final_offer),
                "mandate_violation": not mandate_ok(agent_role, final_offer),
                "target_met": utility(agent_role, final_offer) >= TARGET_UTILITY and mandate_ok(agent_role, final_offer),
                **final_offer,
            }
            return rows, final

    final = {
        "trial_id": trial_id,
        "agent_role": agent_role,
        "opponent_role": opponent_role,
        "outcome": "deadline",
        "rounds": MAX_ROUNDS,
        "own_utility": utility(agent_role, incoming_offer),
        "counterpart_utility": utility(opponent_role, incoming_offer),
        "joint_utility": utility(agent_role, incoming_offer) + utility(opponent_role, incoming_offer),
        "mandate_violation": not mandate_ok(agent_role, incoming_offer),
        "target_met": False,
        **incoming_offer,
    }
    return rows, final


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(finals: list[dict]) -> list[dict]:
    out = []
    for role in ["buyer", "seller"]:
        rows = [r for r in finals if r["agent_role"] == role]
        utils = [r["own_utility"] for r in rows]
        out.append(
            {
                "agent_role": role,
                "n": len(rows),
                "agreement_rate": sum(r["outcome"] in {"agent_accept", "chatgpt_accept"} for r in rows) / len(rows),
                "target_met_rate": sum(r["target_met"] for r in rows) / len(rows),
                "mandate_violation_rate": sum(r["mandate_violation"] for r in rows) / len(rows),
                "mean_own_utility": mean(utils),
                "utility_sd": pstdev(utils) if len(utils) > 1 else 0.0,
                "mean_counterpart_utility": mean(r["counterpart_utility"] for r in rows),
                "mean_joint_utility": mean(r["joint_utility"] for r in rows),
                "mean_rounds": mean(r["rounds"] for r in rows),
            }
        )
    return out


def main() -> None:
    rng = random.Random(SEED)
    policy = load_policy()
    if policy:
        print(f"Loaded trained policy from {POLICY_PATH}", flush=True)
    else:
        print(f"No trained policy found at {POLICY_PATH}; using conservative fallback actions.", flush=True)
    all_rounds: list[dict] = []
    finals: list[dict] = []
    roles = ["buyer", "seller"] * ((TOTAL_TRIALS + 1) // 2)
    for trial_id, role in enumerate(roles[:TOTAL_TRIALS], start=1):
        print(f"Running trial {trial_id}/{TOTAL_TRIALS}: agent_role={role}, model={MODEL}", flush=True)
        rounds, final = run_trial(trial_id, role, rng, policy)
        all_rounds.extend(rounds)
        finals.append(final)
        time.sleep(0.25)

    summary = summarize(finals)
    write_csv(OUT_DIR / "api_experiment_rounds.csv", all_rounds)
    write_csv(OUT_DIR / "api_experiment_final.csv", finals)
    write_csv(OUT_DIR / "api_experiment_summary.csv", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {OUT_DIR / 'api_experiment_summary.csv'}")


if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"OpenAI API HTTP error {exc.code}: {detail}") from exc
