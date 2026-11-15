# Run from grad-master/:  python -m pytest candidate_agent/test_scoring.py
from candidate_agent.agent_tools import CandidateEvaluationAgent

agent = CandidateEvaluationAgent(decision_threshold=6.0)


def candidate(resume=10, quiz=1.0, proctoring=1.0, interview=(1.0, 1.0, 1.0)):
    return {
        "resume_relevance_score": resume,
        "quiz_metrics": [quiz],
        "proctoring_metrics": [proctoring],
        "interview_metrics": list(interview),
    }


def test_perfect_candidate_gets_full_score():
    result = agent.evaluate(candidate())
    assert result["score"] == 10.0
    assert result["decision"] == "Shortlisted"


def test_zero_candidate_gets_zero():
    result = agent.evaluate(candidate(resume=0, quiz=0, proctoring=0, interview=(0, 0, 0)))
    assert result["score"] == 0.0
    assert result["decision"] == "Rejected"


def test_weights_are_normalized_and_interview_dominates():
    result = agent.evaluate(candidate(resume=0, quiz=0, proctoring=0))
    assert abs(sum(result["weights"].values()) - 1) < 0.01
    # interview weight 0.4 out of a total of 0.7
    assert result["score"] == round(10 * 0.4 / 0.7, 2)


def test_lower_proctoring_compliance_lowers_score():
    honest = agent.evaluate(candidate(proctoring=1.0))["score"]
    suspicious = agent.evaluate(candidate(proctoring=0.2))["score"]
    assert suspicious < honest


def test_threshold_is_inclusive():
    assert agent.make_decision(6.0) == "Shortlisted"
    assert agent.make_decision(5.99) == "Rejected"
