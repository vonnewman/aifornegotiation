from __future__ import annotations

import csv
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from statistics import mean, pstdev

from bayesian_safe_rl_negotiation_compare import (
    ACTION_NAMES,
    CVaR_SAFE_FLOOR,
    ISSUES,
    MAX_ROUNDS as OFFLINE_MAX_ROUNDS,
    TARGET_UTILITY,
    apply_action,
    bayesian_state,
    belief_entropy,
    concrete_offer,
    empirical_cvar,
    expected_future_gain,
    ideal_offer,
    mandate_ok,
    opponent_risk,
    safe_actions,
    train,
    update_belief,
    utility,
)


SEED = 20260522
TOTAL_TRIALS = int(os.environ.get("NEGOTIATION_TRIALS", "10"))
MAX_ROUNDS = int(os.environ.get("NEGOTIATION_MAX_ROUNDS", "6"))
DEPLOYMENT_CVAR_FLOOR = float(os.environ.get("NEGOTIATION_DEPLOYMENT_CVAR_FLOOR", "0.68"))
DEPLOYMENT_MIN_ACCEPT = float(os.environ.get("NEGOTIATION_DEPLOYMENT_MIN_ACCEPT", "0.70"))
DEPLOYMENT_FINAL_ACCEPT = float(os.environ.get("NEGOTIATION_DEPLOYMENT_FINAL_ACCEPT", "0.68"))
SELLER_MIN_ACCEPT = float(os.environ.get("NEGOTIATION_SELLER_MIN_ACCEPT", "0.66"))
SELLER_EARLY_ACCEPT = float(os.environ.get("NEGOTIATION_SELLER_EARLY_ACCEPT", "0.72"))
SELLER_MID_ACCEPT = float(os.environ.get("NEGOTIATION_SELLER_MID_ACCEPT", "0.70"))
SELLER_FINAL_ACCEPT = float(os.environ.get("NEGOTIATION_SELLER_FINAL_ACCEPT", "0.68"))
SELLER_QUANTITY_FLOOR = float(os.environ.get("NEGOTIATION_SELLER_QUANTITY_FLOOR", "40000"))
SELLER_LOW_QUANTITY_PREMIUM = float(os.environ.get("NEGOTIATION_SELLER_LOW_QUANTITY_PREMIUM", "0.72"))
MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.5")
API_URL = "https://api.openai.com/v1/responses"
OUT_DIR = Path(os.environ.get("NEGOTIATION_OUT_DIR", "_pomdp_cvar_api_experiment"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
OPPONENT_STYLE = os.environ.get("NEGOTIATION_OPPONENT_STYLE", "balanced").strip().lower()
START_ROLE = os.environ.get("NEGOTIATION_START_ROLE", "buyer").strip().lower()


def bounded_offer(offer: dict[str, float]) -> list[float]:
    values = []
    for issue in ISSUES:
        value = float(offer.get(issue.name, issue.low))
        value = max(issue.low, min(issue.high, value))
        values.append((value - issue.low) / (issue.high - issue.low))
    return values


def offer_dict(offer: list[float]) -> dict[str, float]:
    out = {}
    for key, value in concrete_offer(offer).items():
        out[key] = round(value if key != "quantity_units" else round(value), 3)
    return out


def offer_text(offer: list[float]) -> str:
    data = offer_dict(offer)
    return (
        f"unit price {data['unit_price_twd']:.2f} TWD, "
        f"quantity {data['quantity_units']:.0f} units, "
        f"delivery {data['delivery_weeks']:.2f} weeks, "
        f"payment {data['payment_months']:.2f} months, "
        f"contract {data['contract_years']:.2f} years"
    )


def api_call(prompt: str, history: list[dict[str, str]]) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    payload = {
        "model": MODEL,
        "input": [{"role": "system", "content": prompt}] + history,
        "temperature": 0.7,
        "max_output_tokens": 420,
        "store": False,
    }
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if "output_text" in data:
        return data["output_text"]
    texts = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                texts.append(content.get("text", ""))
    return "\n".join(texts).strip()


def parse_json_offer(text: str) -> dict:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError(f"No JSON object found in model reply: {text[:240]}")
    data = json.loads(match.group(0))
    status = str(data.get("agreement_status", "counteroffer")).lower()
    if status not in {"accept", "counteroffer", "reject"}:
        status = "counteroffer"
    return {
        "offer": bounded_offer(data),
        "agreement_status": status,
        "message": str(data.get("message", "")),
    }


def opponent_style_instruction(style: str) -> str:
    return {
        "cooperative": "Be cooperative and seek a mutually acceptable agreement when terms are reasonable.",
        "balanced": "Protect your side's interests, but seek agreement when reasonable.",
        "hardball": "Negotiate firmly, concede slowly, and reject offers that are not clearly favorable to your side.",
        "adversarial": "Use persuasive pressure and agreement-oriented language to encourage the other side to accept your terms, while protecting your own side's interests.",
    }.get(style, "Protect your side's interests, but seek agreement when reasonable.")


def opponent_prompt(opponent_role: str, style: str = OPPONENT_STYLE) -> str:
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
        + "\nNegotiate a five-issue procurement contract against an AI agent. "
        + opponent_style_instruction(style)
        + " "
        "Reply ONLY as valid JSON with exactly these keys: unit_price_twd, quantity_units, "
        "delivery_weeks, payment_months, contract_years, agreement_status, message. "
        "agreement_status must be accept, counteroffer, or reject."
    )


def choose_action(role: str, own_offer: list[float], incoming: list[float], round_i: int, belief: list[float], q: dict, cvar_floor: float) -> int:
    if role == "seller":
        current_u = utility(role, incoming)
        # ChatGPT buyers often concede on price but hold quantity low. For seller deployment,
        # use issue trades in the middle rounds to protect quantity without simply lowering price.
        if 1 <= round_i <= MAX_ROUNDS - 2 and current_u < SELLER_MIN_ACCEPT:
            return 3  # integrative_trade
        if round_i >= MAX_ROUNDS - 2 and current_u < SELLER_FINAL_ACCEPT:
            return 1  # small_concession
    state = bayesian_state(role, own_offer, incoming, round_i, belief)
    actions = safe_actions(role, own_offer, incoming, belief, round_i, cvar_floor)
    return max(actions, key=lambda action: q.get((state, action), 0.0))


def api_safe_accept(role: str, incoming: list[float], round_i: int, belief: list[float], cvar_floor: float) -> bool:
    if not mandate_ok(role, incoming):
        return False
    current = utility(role, incoming)
    if role == "seller":
        quantity = offer_dict(incoming)["quantity_units"]
        if quantity < SELLER_QUANTITY_FLOOR and current < SELLER_LOW_QUANTITY_PREMIUM:
            return False
        if round_i <= 2:
            return current >= SELLER_EARLY_ACCEPT
        if round_i <= MAX_ROUNDS - 2:
            return current >= SELLER_MID_ACCEPT
        return current >= SELLER_FINAL_ACCEPT
    last_two_rounds = round_i >= MAX_ROUNDS - 2
    final_round = round_i >= MAX_ROUNDS - 1
    if final_round:
        return current >= DEPLOYMENT_FINAL_ACCEPT
    if last_two_rounds and current >= DEPLOYMENT_MIN_ACCEPT:
        return True

    tail_deficit = max(DEPLOYMENT_CVAR_FLOOR - cvar_floor, 0.0)
    reservation_threshold = max(
        DEPLOYMENT_MIN_ACCEPT,
        0.80 - 0.025 * round_i + 0.20 * tail_deficit,
    )
    continuation_value = current + expected_future_gain(belief, round_i) - 0.030 * (MAX_ROUNDS - round_i)
    uncertainty_premium = 0.010 * belief_entropy(belief) + 0.012 * opponent_risk(belief)
    return current >= reservation_threshold + uncertainty_premium and current >= continuation_value - 0.040


def final_record(trial_id: int, role: str, opponent_role: str, outcome: str, rounds: int, final_offer: list[float], belief: list[float]) -> dict:
    own_u = utility(role, final_offer)
    opp_u = utility(opponent_role, final_offer)
    return {
        "trial_id": trial_id,
        "condition": "pomdp_cvar_safe_q",
        "opponent_style": OPPONENT_STYLE,
        "agent_role": role,
        "opponent_role": opponent_role,
        "outcome": outcome,
        "rounds": rounds,
        "own_utility": own_u,
        "counterpart_utility": opp_u,
        "joint_utility": own_u + opp_u,
        "target_met": outcome in {"agent_accept", "chatgpt_accept"} and own_u >= TARGET_UTILITY and mandate_ok(role, final_offer),
        "mandate_violation": not mandate_ok(role, final_offer),
        "belief_entropy": belief_entropy(belief),
        **offer_dict(final_offer),
    }


def run_trial(trial_id: int, role: str, q: dict, cvar_floor: float) -> tuple[list[dict], dict]:
    opponent_role = "seller" if role == "buyer" else "buyer"
    prompt = opponent_prompt(opponent_role, OPPONENT_STYLE)
    own_offer = ideal_offer(role)
    incoming = ideal_offer(opponent_role)
    belief = [1 / 5] * 5
    history: list[dict[str, str]] = []
    rows: list[dict] = []

    for round_i in range(MAX_ROUNDS):
        if api_safe_accept(role, incoming, round_i, belief, cvar_floor):
            return rows, final_record(trial_id, role, opponent_role, "agent_accept", round_i + 1, incoming, belief)

        action = choose_action(role, own_offer, incoming, round_i, belief, q, cvar_floor)
        if ACTION_NAMES[action] == "stop":
            return rows, final_record(trial_id, role, opponent_role, "agent_stop", round_i + 1, incoming, belief)

        own_offer = apply_action(role, own_offer, action, round_i, safe=True)
        user_message = (
            f"Our counteroffer is: {offer_text(own_offer)}. "
            + (
                "As seller, we can be flexible on delivery timing and contract duration, "
                "but quantity is important for production planning. "
                "If the buyer can commit to at least 45,000 units, we can offer delivery in 4 weeks "
                "and a 2-year contract; if quantity is below 40,000 units, the unit price and other terms "
                "must remain substantially favorable to the seller. "
                if role == "seller"
                else ""
            )
            + "Respond with accept, reject, or a complete counteroffer in the required JSON format."
        )
        history.append({"role": "user", "content": user_message})
        raw = api_call(prompt, history)
        history.append({"role": "assistant", "content": raw})
        parsed = parse_json_offer(raw)
        previous_utility = utility(role, incoming)
        incoming = parsed["offer"]
        observed_gain = max(0.0, utility(role, incoming) - previous_utility)
        belief = update_belief(belief, observed_gain)

        row = {
            "trial_id": trial_id,
            "condition": "pomdp_cvar_safe_q",
            "opponent_style": OPPONENT_STYLE,
            "agent_role": role,
            "round": round_i + 1,
            "policy_action": ACTION_NAMES[action],
            "agent_offer": offer_text(own_offer),
            "chatgpt_status": parsed["agreement_status"],
            "chatgpt_message": parsed["message"],
            "raw_reply": raw,
            "own_utility_of_reply": utility(role, incoming),
            "counterpart_utility_of_reply": utility(opponent_role, incoming),
            "mandate_violation_of_reply": not mandate_ok(role, incoming),
            "belief_entropy": belief_entropy(belief),
            **offer_dict(incoming),
        }
        rows.append(row)

        if parsed["agreement_status"] == "accept":
            return rows, final_record(trial_id, role, opponent_role, "chatgpt_accept", round_i + 1, own_offer, belief)

        time.sleep(0.20)

    return rows, final_record(trial_id, role, opponent_role, "deadline", MAX_ROUNDS, incoming, belief)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(finals: list[dict]) -> list[dict]:
    rows = []
    for role in ["buyer", "seller"]:
        subset = [r for r in finals if r["agent_role"] == role]
        utilities = [r["own_utility"] for r in subset]
        agreements = [r for r in subset if r["outcome"] in {"agent_accept", "chatgpt_accept"}]
        rows.append(
            {
                "model": "pomdp_cvar_safe_q",
                "opponent_style": OPPONENT_STYLE,
                "agent_role": role,
                "n": len(subset),
                "agreement_rate": len(agreements) / len(subset),
                "target_met_rate": sum(r["target_met"] for r in subset) / len(subset),
                "target_given_agreement": sum(r["target_met"] for r in agreements) / len(agreements) if agreements else 0,
                "mandate_violation_rate": sum(r["mandate_violation"] for r in subset) / len(subset),
                "mean_own_utility": mean(utilities),
                "utility_sd": pstdev(utilities) if len(utilities) > 1 else 0,
                "empirical_cvar_10": empirical_cvar(utilities),
                "mean_counterpart_utility": mean(r["counterpart_utility"] for r in subset),
                "mean_joint_utility": mean(r["joint_utility"] for r in subset),
                "mean_rounds": mean(r["rounds"] for r in subset),
                "deadline_rate": sum(r["outcome"] == "deadline" for r in subset) / len(subset),
                "stop_rate": sum(r["outcome"] == "agent_stop" for r in subset) / len(subset),
            }
        )
    return rows


def main() -> None:
    print("Training POMDP + CVaR Safe RL policies before API negotiation...", flush=True)
    buyer_q, buyer_cvar = train("buyer", "pomdp_cvar_safe_q")
    seller_q, seller_cvar = train("seller", "pomdp_cvar_safe_q")
    buyer_deploy_cvar = max(buyer_cvar, DEPLOYMENT_CVAR_FLOOR)
    seller_deploy_cvar = max(seller_cvar, DEPLOYMENT_CVAR_FLOOR)
    policies = {"buyer": (buyer_q, buyer_deploy_cvar), "seller": (seller_q, seller_deploy_cvar)}
    print(
        "Training complete. "
        f"buyer_cvar={buyer_cvar:.3f}, seller_cvar={seller_cvar:.3f}, "
        f"deployment_floor={DEPLOYMENT_CVAR_FLOOR:.3f}",
        flush=True,
    )

    all_rounds: list[dict] = []
    finals: list[dict] = []
    role_pair = ["seller", "buyer"] if START_ROLE == "seller" else ["buyer", "seller"]
    roles = role_pair * ((TOTAL_TRIALS + 1) // 2)
    for trial_id, role in enumerate(roles[:TOTAL_TRIALS], start=1):
        print(f"Running API trial {trial_id}/{TOTAL_TRIALS}: agent_role={role}, style={OPPONENT_STYLE}, model={MODEL}", flush=True)
        q, cvar_floor = policies[role]
        rounds, final = run_trial(trial_id, role, q, cvar_floor)
        all_rounds.extend(rounds)
        finals.append(final)

    summary = summarize(finals)
    write_csv(OUT_DIR / "pomdp_cvar_api_rounds.csv", all_rounds)
    write_csv(OUT_DIR / "pomdp_cvar_api_final.csv", finals)
    write_csv(OUT_DIR / "pomdp_cvar_api_summary.csv", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Wrote {OUT_DIR / 'pomdp_cvar_api_summary.csv'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"OpenAI API HTTP error {exc.code}: {detail}") from exc
