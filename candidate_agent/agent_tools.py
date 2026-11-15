from statistics import fmean


class CandidateEvaluationAgent:
    """Transparent, rule-based scoring of a candidate.

    1. Every component is turned into a 0-1 score where higher is better.
    2. The final score is the weighted average of the components, scaled to 0-10.
    3. The decision is a fixed threshold on that score.

    The LLM never changes the score or the decision; it only explains them.
    """

    def __init__(self,
                 weight_resume_relevance=0.1,
                 weight_quiz=0.1,
                 weight_proctoring=0.1,
                 weight_interview=0.4,
                 decision_threshold=6.0):
        self.weights = {
            "resume_relevance": weight_resume_relevance,
            "quiz": weight_quiz,
            "proctoring": weight_proctoring,
            "interview": weight_interview,
        }
        self.decision_threshold = decision_threshold

    def component_scores(self, candidate_data):
        """Each component on a 0-1 scale (higher is better)."""
        return {
            "resume_relevance": candidate_data["resume_relevance_score"] / 10.0,  # given on 0-10
            "quiz": fmean(candidate_data["quiz_metrics"]),
            "proctoring": fmean(candidate_data["proctoring_metrics"]),
            "interview": fmean(candidate_data["interview_metrics"]),
        }

    def make_decision(self, score):
        return "Shortlisted" if score >= self.decision_threshold else "Rejected"

    def evaluate(self, candidate_data):
        components = self.component_scores(candidate_data)
        # Divide by the weight sum so the score always spans the full 0-10 range.
        total_weight = sum(self.weights.values())
        score = round(10 * sum(components[k] * w for k, w in self.weights.items()) / total_weight, 2)
        return {
            "score": score,
            "decision": self.make_decision(score),
            "component_scores": {k: round(v, 2) for k, v in components.items()},
            # How much each component counted toward the final score (sums to 1)
            "weights": {k: round(w / total_weight, 3) for k, w in self.weights.items()},
        }
