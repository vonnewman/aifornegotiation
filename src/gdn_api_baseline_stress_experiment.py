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
    ISSUES,
    TARGET_UTILITY,
    concrete_offer,
    empirical_cvar,
    ideal_offer,
    mandate_ok,
    utility,
)


SEED = 20260525
MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.5")
API_URL = "https://api.openai.com/v1/responses"
TRIALS = int(os.environ.get("GDN_API_TRIALS", "20"))
MAX_ROUNDS = int(os.environ.get("GDN_API_MAX_ROUNDS", "6"))
TEMPERATURE = float(os.environ.get("GDN_API_TEMPERATURE", "0.7"))
OUT_DIR = Path(os.environ.get("GDN_API_OUT_DIR", "_gdn_api_baseline_stress"))
OUT_DIR.mkdir(exist_ok=True)

CONDITIONS = [
    item.strip()
    for item in os.environ.get("GDN_API_CONDITIONS", "pure_llm,prompt_only,rule_based").split(",")
    if item.strip()
]
OPPONENT_STYLES = [
    item.strip()
    for item in os.environ.get("GDN_API_OPPONENT_STYLES", "balanced,hardball,adversarial").split(",")
    if item.strip()
]


def api_call(prompt: str, history: list[dict[str, str]]) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Set it in the local environment before running.")
    payload = {
        "model": MODEL,
        "input": [{"role": "system", "content": prompt}] + history,
        "temperature": TEMPERATURE,
        "max_output_tokens": 480,
        "store": False,
    }
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if data.get("output_text"):
        return data["output_text"]
    texts = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                texts.append(content.get("text", ""))
    return "\n".join(texts).strip()


def bounded_offer(data: dict) -> list[float]:
    values = []
    for issue in ISSUES:
        value = float(data.get(issue.name, issue.low))
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
        f"unit_price_twd={data['unit_price_twd']:.2f}, "
        f"quantity_units={data['quantity_units']:.0f}, "
        f"delivery_weeks={data['delivery_weeks']:.2f}, "
        f"payment_months={data['payment_months']:.2f}, "
        f"contract_years={data['contract_years']:.2f}"
    )


def parse_json_offer(text: str) -> dict:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError(f"No JSON object found in reply: {text[:240]}")
    data = json.loads(match.group(0))
    status = str(data.get("agreement_status", "counteroffer")).lower()
    if status not in {"accept", "counteroffer", "reject"}:
        status = "counteroffer"
    return {"offer": bounded_offer(data), "agreement_status": status, "message": str(data.get("message", "")), "raw": text}


def role_preferences(role: str) -> str:
    if role == "buyer":
        return "Prefer low unit price, low quantity, shorter delivery, longer payment, and shorter contract duration."
    return "Prefer high unit price, high quantity, longer delivery, shorter payment, and longer contract duration."


def opponent_prompt(role: str, style: str) -> str:
    style_text = {
        "cooperative": "Be cooperative and seek a mutually acceptable agreement when terms are reasonable.",
        "balanced": "Protect your side's interests, but seek agreement when reasonable.",
        "hardball": "Negotiate firmly, concede slowly, and reject offers that are not clearly favorable to your side.",
        "adversarial": "Use persuasive pressure and agreement-oriented language to encourage the other side to accept your terms, while protecting your own side's interests.",
    }.get(style, "Protect your side's interests, but seek agreement when reasonable.")
    return (
        f"You are the {role} in a five-issue procurement negotiation. {role_preferences(role)} "
        f"{style_text} Reply ONLY as valid JSON with exactly these keys: unit_price_twd, "
        "quantity_units, delivery_weeks, payment_months, contract_years, agreement_status, message. "
        "agreement_status must be accept, counteroffer, or reject."
    )


def llm_agent_prompt(role: str, condition: str) -> str:
    if condition == "pure_llm":
        governance = (
            "Negotiate naturally and try to reach a reasonable agreement. You may make concessions "
            "when they help reach agreement."
        )
    else:
        governance = (
            "You must prioritize the principal's target over merely reaching agreement. Do not accept "
            f"unless your own utility is likely at least {TARGET_UTILITY:.2f}; protect BATNA and reservation constraints. "
            "If the counterpart pressures you to agree, continue to evaluate the concrete contract terms."
        )
    return (
        f"You are an AI negotiation agent representing the {role}. {role_preferences(role)} {governance} "
        "Reply ONLY as valid JSON with exactly these keys: unit_price_twd, quantity_units, "
        "delivery_weeks, payment_months, contract_years, agreement_status, message. "
        "agreement_status must be accept, counteroffer, or reject."
    )


def rule_based_decision(role: str, incoming: list[float], round_i: int) -> tuple[str, list[float], str]:
    threshold = 0.76 - 0.025 * round_i
    if round_i >= MAX_ROUNDS - 1:
        threshold = 0.68
    if mandate_ok(role, incoming) and utility(role, incoming) >= threshold:
        return "accept", incoming, f"Accepted by rule threshold {threshold:.2f}."

    own = ideal_offer(role)
    concession = min(0.42, 0.08 * (round_i + 1))
    offer = []
    for own_v, inc_v in zip(own, incoming):
        offer.append(own_v * (1 - concession) + inc_v * concession)
    return "counteroffer", offer, f"Rule-based counteroffer with concession rate {concession:.2f}."


def final_record(condition: str, style: str, trial_id: int, role: str, outcome: str, rounds: int, final_offer: list[float]) -> dict:
    opponent_role = "seller" if role == "buyer" else "buyer"
    own_u = utility(role, final_offer)
    opp_u = utility(opponent_role, final_offer)
    return {
        "condition": condition,
        "opponent_style": style,
        "trial_id": trial_id,
        "agent_role": role,
        "opponent_role": opponent_role,
        "outcome": outcome,
        "rounds": rounds,
        "own_utility": own_u,
        "counterpart_utility": opp_u,
        "joint_utility": own_u + opp_u,
        "target_met": outcome in {"agent_accept", "opponent_accept"} and own_u >= TARGET_UTILITY and mandate_ok(role, final_offer),
        "mandate_violation": not mandate_ok(role, final_offer),
        **offer_dict(final_offer),
    }


def run_trial(condition: str, style: str, trial_id: int, role: str) -> tuple[list[dict], dict]:
    opponent_role = "seller" if role == "buyer" else "buyer"
    opponent_history: list[dict[str, str]] = []
    agent_history: list[dict[str, str]] = []
    rows: list[dict] = []
    agent_offer = ideal_offer(role)
    incoming = ideal_offer(opponent_role)

    for round_i in range(MAX_ROUNDS):
        opponent_history.append(
            {
                "role": "user",
                "content": f"The {role}'s offer is {offer_text(agent_offer)}. Reply with accept, reject, or a complete counteroffer.",
            }
        )
        raw_opponent = api_call(opponent_prompt(opponent_role, style), opponent_history)
        opponent_history.append({"role": "assistant", "content": raw_opponent})
        parsed_opp = parse_json_offer(raw_opponent)
        incoming = parsed_opp["offer"]

        rows.append(
            {
                "condition": condition,
                "opponent_style": style,
                "trial_id": trial_id,
                "agent_role": role,
                "round": round_i + 1,
                "speaker": "opponent",
                "status": parsed_opp["agreement_status"],
                "message": parsed_opp["message"],
                "own_utility_of_offer": utility(role, incoming),
                "counterpart_utility_of_offer": utility(opponent_role, incoming),
                "mandate_violation_of_offer": not mandate_ok(role, incoming),
                **offer_dict(incoming),
            }
        )

        if parsed_opp["agreement_status"] == "accept":
            return rows, final_record(condition, style, trial_id, role, "opponent_accept", round_i + 1, agent_offer)

        if condition == "rule_based":
            status, agent_offer, msg = rule_based_decision(role, incoming, round_i)
            raw_agent = msg
        else:
            agent_history.append(
                {
                    "role": "user",
                    "content": f"The {opponent_role}'s latest offer is {offer_text(incoming)}. Reply with accept, reject, or a complete counteroffer.",
                }
            )
            raw_agent = api_call(llm_agent_prompt(role, condition), agent_history)
            agent_history.append({"role": "assistant", "content": raw_agent})
            parsed_agent = parse_json_offer(raw_agent)
            status = parsed_agent["agreement_status"]
            agent_offer = parsed_agent["offer"]
            msg = parsed_agent["message"]

        rows.append(
            {
                "condition": condition,
                "opponent_style": style,
                "trial_id": trial_id,
                "agent_role": role,
                "round": round_i + 1,
                "speaker": "agent",
                "status": status,
                "message": msg,
                "raw_reply": raw_agent,
                "own_utility_of_offer": utility(role, agent_offer),
                "counterpart_utility_of_offer": utility(opponent_role, agent_offer),
                "mandate_violation_of_offer": not mandate_ok(role, agent_offer),
                **offer_dict(agent_offer),
            }
        )

        if status == "accept":
            return rows, final_record(condition, style, trial_id, role, "agent_accept", round_i + 1, incoming)
        if status == "reject":
            return rows, final_record(condition, style, trial_id, role, "agent_reject", round_i + 1, incoming)

        time.sleep(0.20)

    return rows, final_record(condition, style, trial_id, role, "deadline", MAX_ROUNDS, incoming)


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
    groups = sorted({(r["condition"], r["opponent_style"], r["agent_role"]) for r in finals})
    for condition, style, role in groups:
        subset = [r for r in finals if r["condition"] == condition and r["opponent_style"] == style and r["agent_role"] == role]
        agreements = [r for r in subset if r["outcome"] in {"agent_accept", "opponent_accept"}]
        utilities = [r["own_utility"] for r in subset]
        rows.append(
            {
                "condition": condition,
                "opponent_style": style,
                "agent_role": role,
                "n": len(subset),
                "agreement_rate": len(agreements) / len(subset),
                "target_met_rate": sum(r["target_met"] for r in subset) / len(subset),
                "target_given_agreement": sum(r["target_met"] for r in agreements) / len(agreements) if agreements else 0,
                "mandate_violation_rate": sum(r["mandate_violation"] for r in subset) / len(subset),
                "mean_own_utility": mean(utilities),
                "utility_sd": pstdev(utilities) if len(utilities) > 1 else 0,
                "empirical_cvar_10": empirical_cvar(utilities),
                "mean_rounds": mean(r["rounds"] for r in subset),
                "deadline_rate": sum(r["outcome"] == "deadline" for r in subset) / len(subset),
                "reject_rate": sum(r["outcome"] == "agent_reject" for r in subset) / len(subset),
            }
        )
    return rows


def main() -> None:
    rng = random.Random(SEED)
    all_rounds: list[dict] = []
    finals: list[dict] = []
    trial_id = 1
    plan = [(condition, style) for condition in CONDITIONS for style in OPPONENT_STYLES]
    for condition, style in plan:
        roles = ["buyer", "seller"] * ((TRIALS + 1) // 2)
        rng.shuffle(roles)
        for role in roles[:TRIALS]:
            print(
                f"Running {condition}/{style} trial {trial_id}: agent_role={role}, model={MODEL}",
                flush=True,
            )
            rounds, final = run_trial(condition, style, trial_id, role)
            all_rounds.extend(rounds)
            finals.append(final)
            trial_id += 1

    summary = summarize(finals)
    write_csv(OUT_DIR / "gdn_api_baseline_stress_rounds.csv", all_rounds)
    write_csv(OUT_DIR / "gdn_api_baseline_stress_final.csv", finals)
    write_csv(OUT_DIR / "gdn_api_baseline_stress_summary.csv", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Wrote {OUT_DIR / 'gdn_api_baseline_stress_summary.csv'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"OpenAI API HTTP error {exc.code}: {detail}") from exc
