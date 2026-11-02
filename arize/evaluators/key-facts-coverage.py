class KeyFactsCoverage(CodeEvaluator):
    """Share of the semicolon-separated key facts found in the answer ("A OR B" = either is enough)."""

    STOP = {"the", "a", "an", "of", "in", "on", "at", "and", "is", "are", "to", "for", "with", "by", "as"}

    def _tokens(self, text):
        return [
            t
            for t in re.findall(r"[a-z0-9]+", str(text).lower().replace("'", "").replace("\u2019", "").replace("\u2018", ""))
            if t not in self.STOP
        ]

    def evaluate(
        self,
        *,
        output: Optional[str] = None,
        key_facts: Optional[str] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        facts = [f.strip() for f in str(key_facts or "").replace("None", "").split(";") if f.strip()]
        if not facts:
            return EvaluationResult(label="not_applicable", score=None, explanation="row has no key facts (decline row)")
        answer = set(self._tokens(output or ""))
        missed = []
        for fact in facts:
            alternatives = [a.strip() for a in re.split(r"\s+OR\s+", fact) if a.strip()]
            found = False
            for alt in alternatives:
                words = self._tokens(alt)
                if words and sum(w in answer for w in words) / len(words) >= 0.7:
                    found = True
                    break
            if not found:
                missed.append(fact)
        score = (len(facts) - len(missed)) / len(facts)
        label = "all_present" if not missed else ("partial" if len(missed) < len(facts) else "none_present")
        return EvaluationResult(
            label=label,
            score=round(score, 2),
            explanation=("missing: " + "; ".join(missed)) if missed else "all key facts present",
        )
