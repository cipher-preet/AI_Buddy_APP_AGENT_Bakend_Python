from __future__ import annotations

from typing import Any

from services.goals.agents.llm import call_goal_agent, goal_context_payload
from services.goals.domains import detect_goal_type
from services.goals.schemas import (
    GoalAdvisorResult,
    GoalCriticResult,
    GoalExecutionResult,
    GoalStrategistResult,
)


LAUNCH_GOAL_TYPES = {"product_launch", "app_build"}


def _text(value: Any) -> str:
    return str(value or "").strip()


def wants_expert_pipeline(goal: dict[str, Any]) -> bool:
    """Every goal gets the full Strategist → Advisor → Execution → Critic treatment."""
    return True


def is_launch_style_goal(goal: dict[str, Any]) -> bool:
    """Used only to pick a launch-shaped fallback when agents fail."""
    from services.goals.domains import normalize_domain, understanding_as_dict

    understanding = understanding_as_dict(goal.get("understanding"))
    domain = normalize_domain(str(understanding.get("domain") or ""))
    if domain in {"CAREER", "EDUCATION", "HEALTH", "PRODUCTIVITY", "FINANCIAL", "TRAVEL", "RELATIONSHIPS"}:
        return False
    goal_type = detect_goal_type(
        _text(goal.get("statement")),
        _text(goal.get("goalType")),
        domain,
    )
    blob = f"{_text(goal.get('statement'))} {_text(goal.get('title'))} {_text(goal.get('description'))}".lower()
    if goal_type == "product_launch":
        return True
    if domain in {"BUSINESS"} and any(token in blob for token in ("saas", "startup", "public beta", "launch a")):
        return True
    return False


def _coverage_allows(new_count: int, old_count: int, goal: dict[str, Any]) -> bool:
    """Do not let a critic rewrite shrink a long goal back to one phase."""
    from services.goals.brief import stated_deadline

    if new_count <= 0:
        return False
    if old_count >= 4 and new_count <= 1:
        return False
    stated = stated_deadline(goal)
    if stated and stated[0] >= 45 and new_count < 4 and new_count <= max(old_count, 1):
        return False
    return True


def _repairs_are_safe(critic: GoalCriticResult) -> bool:
    """Ignore a critic rewrite that is empty or falls back to generic filler."""
    from services.goals.domains import GENERIC_TASK_PHRASES

    chunks: list[str] = []
    for day in critic.repairedDays or []:
        chunks.append(_text(day.title))
        chunks.extend(_text(task.text) for task in (day.tasks or []))
    for strategy in critic.repairedStrategies or []:
        chunks.append(_text(strategy.title))
        chunks.append(_text(strategy.body))
    for item in critic.repairedPriorityRecommendations or []:
        chunks.append(_text(item.title))
        chunks.append(_text(item.reason))
    for milestone in critic.repairedMilestones or []:
        chunks.append(_text(milestone.title))
    blob = " ".join(chunks).lower()
    if not blob.strip():
        return False
    hits = sum(1 for phrase in GENERIC_TASK_PHRASES if phrase in blob)
    return hits < 2


def _dump_model(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    return {}


async def run_expert_plan_pipeline(goal: dict[str, Any]) -> dict[str, Any] | None:
    """Strategist → Advisor → Execution → Critic, then the scheduler sets dates."""
    try:
        return await _run_expert_plan_pipeline(goal)
    except Exception as error:
        print(
            "goal expert pipeline failed",
            {"error": type(error).__name__, "detail": str(error)[:240]},
            flush=True,
        )
        return None


async def _run_expert_plan_pipeline(goal: dict[str, Any]) -> dict[str, Any] | None:
    context = goal_context_payload(goal)
    models: list[str] = []

    strategist_call = await call_goal_agent(
        prompt_name="goal-strategist-v1",
        schema=GoalStrategistResult,
        payload=context,
        stage="goal_strategist",
        temperature=0.2,
        max_tokens=2200,
    )
    if strategist_call is None:
        return None
    strategy, strategy_model = strategist_call
    models.append(f"strategist:{strategy_model}")

    advisor_payload = {
        **context,
        "strategy": _dump_model(strategy),
    }
    advisor_call = await call_goal_agent(
        prompt_name="goal-advisor-v1",
        schema=GoalAdvisorResult,
        payload=advisor_payload,
        stage="goal_advisor",
        temperature=0.2,
        max_tokens=2800,
    )
    advisor: GoalAdvisorResult | None = None
    if advisor_call is not None:
        advisor, advisor_model = advisor_call
        models.append(f"advisor:{advisor_model}")
        if _text(advisor.positioningHeadline) and not _text(strategy.positioningHeadline):
            strategy.positioningHeadline = advisor.positioningHeadline
        if _text(advisor.positioningBody) and not _text(strategy.positioningBody):
            strategy.positioningBody = advisor.positioningBody

    execution_payload = {
        **context,
        "strategy": _dump_model(strategy),
        "advisor": _dump_model(advisor),
    }
    execution_call = await call_goal_agent(
        prompt_name="goal-execution-v1",
        schema=GoalExecutionResult,
        payload=execution_payload,
        stage="goal_execution",
        temperature=0.15,
        max_tokens=4500,
    )
    if execution_call is None:
        return {
            "strategy": strategy,
            "advisor": advisor,
            "execution": None,
            "critic": None,
            "models": models,
        }
    execution, execution_model = execution_call
    models.append(f"execution:{execution_model}")

    # Prefer advisor strategies/recs when execution left them thin.
    if advisor is not None:
        if (not execution.strategies or len(execution.strategies) < 3) and advisor.strategies:
            execution.strategies = advisor.strategies
        if (
            not execution.priorityRecommendations or len(execution.priorityRecommendations) < 3
        ) and advisor.priorityRecommendations:
            execution.priorityRecommendations = advisor.priorityRecommendations

    critic_payload = {
        **context,
        "strategy": _dump_model(strategy),
        "advisor": _dump_model(advisor),
        "execution": _dump_model(execution),
    }
    critic_call = await call_goal_agent(
        prompt_name="goal-critic-v1",
        schema=GoalCriticResult,
        payload=critic_payload,
        stage="goal_critic",
        temperature=0.1,
        max_tokens=2800,
    )
    critic: GoalCriticResult | None = None
    if critic_call is not None:
        critic, critic_model = critic_call
        models.append(f"critic:{critic_model}")
        if (
            not critic.accepted
            and _repairs_are_safe(critic)
            and (
                critic.repairedDays
                or critic.repairedStrategies
                or critic.repairedPriorityRecommendations
                or critic.repairedChecklists
                or critic.repairedMilestones
            )
        ):
            if critic.repairedDays and _coverage_allows(len(critic.repairedDays), len(execution.days or []), goal):
                execution.days = critic.repairedDays
            if critic.repairedStrategies:
                execution.strategies = critic.repairedStrategies
            if critic.repairedPriorityRecommendations:
                execution.priorityRecommendations = critic.repairedPriorityRecommendations
            if critic.repairedChecklists:
                execution.checklists = critic.repairedChecklists
            if critic.repairedScorecard:
                execution.scorecard = critic.repairedScorecard
            if critic.repairedMilestones and _coverage_allows(
                len(critic.repairedMilestones), len(execution.milestones or []), goal
            ):
                execution.milestones = critic.repairedMilestones
            if critic.rewriteSummary:
                execution.summary = critic.rewriteSummary
            if critic.rewriteRecommendation:
                strategy.recommendation = critic.rewriteRecommendation
        models.append("scheduler:dates-and-steps")

    if "scheduler:dates-and-steps" not in models:
        models.append("scheduler:dates-and-steps")

    return {
        "strategy": strategy,
        "advisor": advisor,
        "execution": execution,
        "critic": critic,
        "models": models,
    }
