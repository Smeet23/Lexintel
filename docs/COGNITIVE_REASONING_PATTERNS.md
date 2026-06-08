# Cognitive Reasoning Patterns: Technical Reference for Implementation

**For Lexintel Developers**
**March 2026**

---

## Overview

This document provides specific, implementable patterns for each cognitive layer identified in the lawyer cognition research. Use these as templates for your architecture redesign.

---

## Pattern 1: Problem Formulation Engine

### What It Does
Transforms vague client facts into structured legal issues.

### Input
```
Client facts (free text): "My partner stopped communicating and won't let me see
company financials. We started the business together 5 years ago. I've put in
$200k. He's refusing to buy me out or show where the money went."
```

### Process

**Step 1: Extract Factual Elements**
```python
def extract_facts(client_input: str) -> List[Fact]:
    """
    Extract legally-relevant facts from narrative.
    """
    facts = [
        Fact(type="parties", value="business partner relationship"),
        Fact(type="timeline", value="5 years together"),
        Fact(type="financial_stake", value="$200k investment"),
        Fact(type="trigger_event", value="refusal to communicate"),
        Fact(type="key_action", value="withholding financial information"),
        Fact(type="remedy_sought", value="buy-out or transparency"),
    ]
    return facts
```

**Step 2: Identify Legal Triggers**
```python
def identify_legal_triggers(facts: List[Fact]) -> List[str]:
    """
    Map facts to legal concepts.

    Withholding info + partner relationship → Fiduciary Duty
    Investment + refusal to exit → Shareholder Oppression
    Etc.
    """
    triggers = [
        "fiduciary_duty_breach",
        "shareholder_oppression",
        "access_to_information_rights",
        "breach_of_partnership_agreement",
        "breach_of_contract_implied"
    ]
    return triggers
```

**Step 3: Map to Legal Issues**
```python
def formulate_legal_issues(
    triggers: List[str],
    business_structure: str = None
) -> IssueFormulation:
    """
    Convert legal triggers to formal legal issues.
    """

    # Rank by likelihood and importance
    primary = LegalIssue(
        name="Fiduciary Duty Breach",
        description="Did partner violate duty to disclose financial information?",
        confidence=0.85,
        jurisdiction_factors=["state_law", "entity_type"],
        evidence_requirements=["partnership_agreement", "communications", "financial_docs"]
    )

    secondary = [
        LegalIssue(
            name="Shareholder Oppression",
            description="Is partner's conduct oppressive to minority shareholder?",
            confidence=0.60 if business_structure == "corporation" else 0.10,
            applies_if="corporation or LLC"
        ),
        LegalIssue(
            name="Access to Information Rights",
            description="Does partner have legal right to inspect books/records?",
            confidence=0.70,
            depends_on=["entity_type", "state_statute"]
        )
    ]

    return IssueFormulation(
        primary_issue=primary,
        secondary_issues=secondary,
        legal_domains=["corporate_law", "partnership_law", "fiduciary_duty"],
        analysis_priority=["primary", "secondary_access_rights", "secondary_oppression"]
    )
```

### Output
```json
{
  "primary_issue": {
    "name": "Fiduciary Duty Breach",
    "legal_question": "Did partner breach fiduciary duty to disclose financial information?",
    "confidence": 0.85,
    "key_factors": [
      "partnership/entity relationship creates fiduciary duty",
      "duty includes disclosure of material financial information",
      "refusal to provide access likely violates duty",
      "damages available if breach proven"
    ]
  },
  "secondary_issues": [
    {
      "name": "Access to Information Rights",
      "confidence": 0.70,
      "depends_on": "state statute for entity type"
    },
    {
      "name": "Shareholder Oppression",
      "confidence": 0.15,
      "only_applies": "if corporation/LLC"
    }
  ],
  "legal_domains": ["fiduciary_duty", "partnership_law", "corporate_governance"],
  "next_research_focus": "fiduciary_duty",
  "missing_info_needed": ["entity type", "state jurisdiction", "partnership agreement"]
}
```

### Implementation Tips

1. **Use a mapping table** for triggers → legal issues
   ```python
   TRIGGER_MAP = {
       ("withholding_info", "partner"): "fiduciary_duty_breach",
       ("business_exit_refusal", "minority_shareholder"): "oppression",
       ("buy_out_refusal", "shareholder"): "forced_buyout_claim",
       # ... more mappings
   }
   ```

2. **Make it iterative**: User can refine. "Does the entity structure matter?" "Yes/No" → updates confidence scores

3. **Track missing information**: Flag things you need to know to formulate better

4. **Domain-specific**: Build separate maps for different practice areas (contract, corporate, tort, etc.)

---

## Pattern 2: Multi-Hypothesis Generation

### What It Does
Generates competing legal interpretations with confidence scores.

### Input
```
Legal issue: Fiduciary duty breach - disclosure of financial information
Relevant cases: [Smith v. Jones, Acme v. XYZ, Baker Corp., ...]
Facts: Partner withholding financials, 5-year partnership, $200k investment
```

### Process

**Step 1: Generate Hypotheses**
```python
def generate_hypotheses(
    legal_issue: str,
    relevant_cases: List[Document],
    facts: str
) -> List[LegalHypothesis]:
    """
    Generate competing interpretations of how law applies.
    """

    hypotheses = [
        LegalHypothesis(
            id="H1",
            name="Strict Disclosure Requirement",
            interpretation="""
            Partners have absolute fiduciary duty to disclose all financial
            information upon request. Refusal is per se breach.
            """,
            supporting_elements=[
                "Partnership statute §8.01 requires disclosure",
                "Smith v. Jones (binding precedent) establishes strict standard",
                "Policy: Partners as agents require full transparency"
            ],
            confidence_baseline=0.65
        ),
        LegalHypothesis(
            id="H2",
            name="Qualified Disclosure Duty",
            interpretation="""
            Partners have duty to disclose material information but only
            if request is reasonable and for legitimate partnership purpose.
            """,
            supporting_elements=[
                "Acme v. XYZ (persuasive authority) suggests qualifications",
                "UPA § 403 doesn't require disclosure for every inquiry",
                "Policy: Protect partners' privacy in sensitive matters"
            ],
            confidence_baseline=0.25
        ),
        LegalHypothesis(
            id="H3",
            name="No Mandatory Disclosure",
            interpretation="""
            Partners have no affirmative duty to disclose. Only if partner
            specifically asks and has legitimate reason.
            """,
            supporting_elements=[
                "Baker Corp. (minority view) suggests minimal obligation",
                "Policy: Caveat emptor - partners should have negotiated",
                "No recent case supports this in your jurisdiction"
            ],
            confidence_baseline=0.10
        )
    ]

    return hypotheses
```

**Step 2: Evaluate Evidence**
```python
def evaluate_hypothesis(
    hypothesis: LegalHypothesis,
    relevant_cases: List[Document],
    facts: str
) -> EvaluatedHypothesis:
    """
    Score hypothesis based on evidence.
    """

    supporting_cases = find_supporting_cases(hypothesis, relevant_cases)
    opposing_cases = find_opposing_cases(hypothesis, relevant_cases)

    # Calculate scores
    precedent_score = len(supporting_cases) / (len(supporting_cases) + len(opposing_cases))

    temporal_score = evaluate_recency(supporting_cases)
    # Recent cases should count more

    authority_score = evaluate_court_level(supporting_cases)
    # Supreme court > appellate > trial

    policy_score = evaluate_policy_alignment(hypothesis, facts)

    # Combine scores
    final_confidence = weighted_average(
        precedent=precedent_score * 0.40,
        temporal=temporal_score * 0.20,
        authority=authority_score * 0.25,
        policy=policy_score * 0.15
    )

    return EvaluatedHypothesis(
        hypothesis=hypothesis,
        supporting_cases=supporting_cases,
        opposing_cases=opposing_cases,
        confidence_adjusted=final_confidence,
        confidence_change=final_confidence - hypothesis.confidence_baseline,
        reasoning=[
            f"Supporting precedents: {len(supporting_cases)}",
            f"Opposing precedents: {len(opposing_cases)}",
            f"Strongest supporting case: {supporting_cases[0].name if supporting_cases else 'none'}",
            f"Main weakness: {identify_main_weakness(hypothesis, opposing_cases)}"
        ]
    )
```

**Step 3: Generate Alternatives**
```python
def identify_weaknesses(
    hypothesis: EvaluatedHypothesis
) -> List[Weakness]:
    """
    What could go wrong with this interpretation?
    """
    weaknesses = [
        Weakness(
            type="temporal",
            description="Recent appellate decisions suggest trend away from strict standard",
            severity="moderate"
        ),
        Weakness(
            type="factual",
            description="Your facts don't perfectly match Smith v. Jones (their disclosure was different)",
            severity="low"
        ),
        Weakness(
            type="policy",
            description="Modern trend values privacy; courts may not enforce full disclosure",
            severity="moderate"
        )
    ]
    return weaknesses
```

### Output
```json
{
  "hypotheses": [
    {
      "id": "H1",
      "name": "Strict Disclosure Requirement",
      "confidence": 0.68,
      "explanation": "Based on binding precedent + policy rationale",
      "supporting_precedents": [
        "Smith v. Jones (2020) - binding, same fact pattern",
        "State Supreme Court (2018) - established strict standard",
        "Acme v. XYZ (2015) - reinforced obligation"
      ],
      "opposing_arguments": [
        "Baker Corp. (2015) - minority view allows qualifications",
        "Federal court (2022) - different jurisdiction, but suggests trend"
      ],
      "key_weakness": "Recent cases hint at narrowing the standard",
      "if_correct": "Partner's refusal is breach; you have damages claim",
      "if_wrong": "You may need to prove you had legitimate reason to ask"
    },
    {
      "id": "H2",
      "name": "Qualified Disclosure Duty",
      "confidence": 0.27,
      "explanation": "Partner must disclose if you show legitimate partnership purpose",
      "supporting_precedents": ["One appellate case suggests this approach"],
      "opposing_arguments": ["Contradicts binding precedent", "Your state rejected this"],
      "risk_level": "moderate"
    },
    {
      "id": "H3",
      "name": "No Mandatory Disclosure",
      "confidence": 0.05,
      "explanation": "Partner owes no affirmative duty",
      "supporting_precedents": ["No recent cases support this"],
      "risk_level": "extremely high if true"
    }
  ],
  "overall_confidence": "moderate (68%) - primary interpretation is likely but not certain",
  "key_uncertainty": "Whether recent trends narrowing disclosure duties will apply",
  "next_research": "Check latest appellate decisions from past 12 months"
}
```

### Implementation Tips

1. **Confidence is Bayesian**: Update based on evidence
   - Start with prior (basic plausibility): 0.33 (if three hypotheses)
   - Update with precedents: How many cases support this?
   - Update with recency: Are supporting cases recent or old?
   - Update with policy: Does doctrine trend toward or away from this?

2. **Always include 3+ hypotheses**: Even if one is clearly strongest, show alternatives

3. **Make confidence transparent**: Show *why* you're 68% not 70%

4. **Track doctrine shifts**: If all recent cases contradict older precedent, note it

---

## Pattern 3: Schema-Based Result Organization

### What It Does
Organizes retrieved documents by legal structure, not just relevance ranking.

### Input
```
Legal issue: "Fiduciary duty disclosure requirement"
Retrieved documents: [20 cases, 5 statutes, 3 law review articles, ...]
```

### Process

**Step 1: Classify Documents**
```python
def classify_document(
    doc: Document,
    legal_issue: str,
    schema: LegalSchema
) -> DocumentCategory:
    """
    Classify document by legal schema.
    """

    categories = {
        "core_holding": "Directly addresses whether disclosure duty exists",
        "supporting_reasoning": "Explains rationale behind the rule",
        "secondary_issue": "Addresses related but not primary issue",
        "opposing_authority": "Contradicts or limits the rule",
        "distinguishable": "Factually different; doesn't apply",
        "policy_rationale": "Why law should be this way",
        "statutory_framework": "Relevant statute or regulation",
        "historical_development": "How doctrine evolved"
    }

    # Classify document
    category = classify_by_content(doc, categories, legal_issue)

    # Score authority level
    authority = score_authority(
        court_level=doc.court,
        recency=doc.date,
        jurisdiction=doc.jurisdiction,
        binding_vs_persuasive=doc.binding_status
    )

    return DocumentCategory(
        doc=doc,
        category=category,
        authority_score=authority,
        relevance_to_issue=doc.relevance_score
    )
```

**Step 2: Organize by Schema**
```python
def organize_by_schema(
    classified_docs: List[DocumentCategory],
    legal_issue: str
) -> SchemaOrganizedResults:
    """
    Organize documents by legal structure.
    """

    organized = {
        "binding_precedent": [],
        "persuasive_precedent": [],
        "supporting_reasoning": [],
        "opposing_authority": [],
        "statutory_foundation": [],
        "policy_arguments": [],
        "distinguishable_cases": [],
        "secondary_issues": []
    }

    # Categorize
    for classified in classified_docs:
        category = classified.category
        organized[category].append(classified)

    # Within each category, sort by authority
    for category in organized:
        organized[category].sort(
            key=lambda x: x.authority_score,
            reverse=True
        )

    return SchemaOrganizedResults(
        binding_precedent=organized["binding_precedent"][:5],
        persuasive_precedent=organized["persuasive_precedent"][:5],
        supporting_reasoning=organized["supporting_reasoning"][:5],
        # ... etc
    )
```

**Step 3: Generate Summary**
```python
def summarize_schema_results(
    organized: SchemaOrganizedResults
) -> SchemaVisualization:
    """
    Create readable visualization of legal landscape.
    """

    summary = f"""
LEGAL LANDSCAPE: Fiduciary Duty Disclosure Requirement
═══════════════════════════════════════════════════════

BINDING PRECEDENT (You Must Follow)
├─ Smith v. Jones (State Supreme, 2020)
│  "Partners must disclose financial information upon reasonable request"
│  Direct application to your facts
│
└─ Acme v. XYZ (State Court of Appeals, 2018)
   "Refusal to disclose material information is per se breach"
   Strong supporting authority

PERSUASIVE PRECEDENT (Influential But Not Required)
├─ Baker Corp. (Federal Court, 2015)
│  "Suggests narrowing the disclosure standard"
│  Different jurisdiction; trend indicator
│
└─ Tech Co. v. Supplier (Neighboring State, 2022)
   "Disclosure duty includes email/digital records"
   Modern jurisdiction, newer approach

OPPOSING AUTHORITY (What Defendant Will Argue)
└─ Minority view (3 cases nationally)
   "Disclosure only if legitimate partnership purpose shown"
   Rejected in your state; weak argument but exists

STATUTORY FOUNDATION
├─ Partnership Act § 8.01
│  "Partners have right to access partnership books and records"
│
└─ Model UPA § 403
   "Access upon request for information related to business"

POLICY RATIONALE
├─ Partnership is fiduciary relationship requiring transparency
├─ Modern business practice expects full financial disclosure
└─ Minority partner protection is important policy

KEY INSIGHT:
Your jurisdiction strongly favors disclosure requirement.
Three binding precedents support your position.
Opposing argument exists but is minority view.
Confidence: 70% likely to succeed if litigated
"""

    return SchemaVisualization(summary=summary)
```

### Output
```
LEGAL LANDSCAPE: Fiduciary Duty Disclosure Requirement

BINDING PRECEDENT (Must Follow)
├─ Smith v. Jones (2020) - Direct precedent, same facts
├─ Acme v. XYZ (2018) - Reinforces strict standard
└─ State Supreme Court Opinion (2019) - Reaffirmed requirement

PERSUASIVE PRECEDENT (Influential)
├─ Baker Corp. (Federal) - Different approach, minority view
└─ 2022 Tech Company Case - Modern interpretation

OPPOSING ARGUMENTS (What defendant will say)
├─ Minority view: Qualification required
├─ Policy argument: Privacy concerns
└─ Factual argument: Not reasonably requested

STATUTORY FRAMEWORK
├─ Partnership Act § 8.01 - Access right
└─ Model UPA § 403 - "Reasonable request"

POLICY RATIONALE
└─ Transparency in fiduciary relationships

OVERALL ASSESSMENT
Binding precedent clearly favors disclosure requirement.
One opposing case exists but is minority/persuasive only.
Statute supports your position.
Likelihood of success: 70%
```

### Implementation Tips

1. **Build category taxonomy** specific to your domains
   ```python
   CATEGORIES = {
       "contract": ["interpretation", "formation", "breach", "remedies"],
       "tort": ["duty", "breach", "causation", "damages"],
       "corporate": ["governance", "fiduciary", "shareholder_rights"]
   }
   ```

2. **Authority scoring**: Not all precedents are equal
   ```python
   def score_authority(court, recency, jurisdiction):
       score = 0
       if court == "supreme_court": score += 0.40
       if court == "appellate": score += 0.25
       if court == "trial": score += 0.10

       # Recency
       if date_diff < 2: score += 0.20  # Very recent
       if date_diff < 5: score += 0.10

       # Jurisdiction
       if jurisdiction == "home_state": score += 0.15
       if jurisdiction == "federal": score += 0.05

       return min(score, 1.0)
   ```

3. **Make it interactive**: User can reorganize if they prefer different schema

---

## Pattern 4: Confidence Scoring Framework

### What It Does
Scores confidence in legal conclusions based on evidence.

### Framework

**Formula**:
```
Confidence = (Precedent_Support × 0.40) +
             (Temporal_Trends × 0.20) +
             (Policy_Alignment × 0.20) +
             (Fact_Fit × 0.20)
```

**Precedent Support** (0-1 scale)
```python
def score_precedent_support(
    supporting_cases: List[Document],
    opposing_cases: List[Document],
    authority_level: str
) -> float:
    """
    How much precedent supports this interpretation?
    """

    if len(supporting_cases) == 0:
        return 0.0

    # Count supporting
    binding_support = len([c for c in supporting_cases if c.is_binding])
    persuasive_support = len([c for c in supporting_cases if not c.is_binding])

    # Count opposing
    binding_oppose = len([c for c in opposing_cases if c.is_binding])
    persuasive_oppose = len([c for c in opposing_cases if not c.is_binding])

    # Score
    support_score = (
        binding_support * 2 +        # Binding counts double
        persuasive_support * 1
    ) / max(
        (binding_oppose * 2 + persuasive_oppose * 1),
        1
    )

    # Cap at 1.0
    return min(support_score, 1.0)
```

**Temporal Trends** (0-1 scale)
```python
def score_temporal_trends(
    cases: List[Document],
    issue: str
) -> float:
    """
    Is doctrine moving toward or away from this interpretation?
    """

    # Trend direction
    recent_10yr = [c for c in cases if year_now() - c.year < 10]
    old_10yr = [c for c in cases if year_now() - c.year >= 10]

    recent_support = len([c for c in recent_10yr if c.supports_interpretation])
    old_support = len([c for c in old_10yr if c.supports_interpretation])

    if len(recent_10yr) == 0:
        return 0.5  # No recent cases

    recent_ratio = recent_support / len(recent_10yr)
    old_ratio = old_support / len(old_10yr) if old_10yr else 0.5

    # If recent > old, trend is positive
    trend_score = recent_ratio - (old_ratio * 0.2)  # Give less weight to old

    return max(0.0, min(trend_score, 1.0))
```

**Policy Alignment** (0-1 scale)
```python
def score_policy_alignment(
    interpretation: str,
    policy_rationale: str,
    opposing_policy: str
) -> float:
    """
    Does policy favor this interpretation?
    """

    # Evaluate which policy argument is stronger
    our_policy_strength = evaluate_policy_strength(policy_rationale)
    their_policy_strength = evaluate_policy_strength(opposing_policy)

    # In modern courts, some policies win over others
    policy_weight = our_policy_strength / (our_policy_strength + their_policy_strength)

    return policy_weight
```

**Fact Fit** (0-1 scale)
```python
def score_fact_fit(
    supporting_cases: List[Document],
    client_facts: str
) -> float:
    """
    How similar are our facts to supporting precedents?
    """

    if not supporting_cases:
        return 0.5  # Unknown

    similarity_scores = []
    for case in supporting_cases:
        similarity = calculate_factual_similarity(
            client_facts,
            case.facts
        )
        similarity_scores.append(similarity)

    # Average similarity to supporting cases
    avg_similarity = sum(similarity_scores) / len(similarity_scores)

    return avg_similarity
```

### Example Output

```
CONFIDENCE ASSESSMENT
═══════════════════════════════════════════════════════════

Interpretation: "Partner must disclose financial information"

Components:
├─ Precedent Support: 0.85
│  (8 supporting cases, 1 opposing case)
│  Binding precedents favor interpretation: 5-1
│
├─ Temporal Trends: 0.75
│  (Recent cases more supportive than old cases)
│  Trend: Courts expanding disclosure duties (favorable)
│
├─ Policy Alignment: 0.80
│  (Transparency in fiduciary relationships is strong policy)
│  Opposing policy (privacy) is weaker
│
└─ Fact Fit: 0.72
   (Your facts closely match Smith v. Jones, less similar to Acme)
   Close match to primary precedent

OVERALL CONFIDENCE: 0.78 (78%)
═══════════════════════════════════════════════════════════

Interpretation: "Disclosure is Conditional" (Alternative)

Components:
├─ Precedent Support: 0.30
│  (1 supporting case, 8 opposing cases)
│  Binding precedents contradict: 1-5
│
├─ Temporal Trends: 0.20
│  (Older cases supported, newer cases don't)
│  Trend is negative
│
├─ Policy Alignment: 0.40
│  (Privacy argument is weak compared to transparency)
│
└─ Fact Fit: 0.25
   (Your facts don't match the one supporting case)

OVERALL CONFIDENCE: 0.26 (26%)
═══════════════════════════════════════════════════════════

INTERPRETATION: Use primary interpretation with 78% confidence.
If court adopts alternative (22% chance), you lose.
Prepare for worst case accordingly.
```

---

## Pattern 5: Narrative Construction

### What It Does
Helps construct persuasive legal narratives from facts and law.

### Process

**Step 1: Identify Key Facts**
```python
def extract_key_facts(
    all_facts: str,
    legal_issue: str,
    interpretation: str
) -> List[KeyFact]:
    """
    Which facts actually matter for this legal conclusion?
    """

    key_facts = [
        KeyFact(
            fact="Partnership formed 5 years ago",
            why_matters="Establishes fiduciary relationship",
            legal_weight=0.8,
            narrative_order=1
        ),
        KeyFact(
            fact="$200k investment by plaintiff",
            why_matters="Shows stake and reliance",
            legal_weight=0.6,
            narrative_order=2,
            emotional_value=0.8  # This matters for jury
        ),
        KeyFact(
            fact="Defendant refused to disclose financials",
            why_matters="Direct breach of disclosure duty",
            legal_weight=1.0,
            narrative_order=3
        ),
        KeyFact(
            fact="Plaintiff lost business opportunity as result",
            why_matters="Damages from breach",
            legal_weight=0.7,
            narrative_order=4,
            emotional_value=0.9
        )
    ]

    return sorted(key_facts, key=lambda x: x.narrative_order)
```

**Step 2: Create Narrative Arc**
```python
def construct_narrative_arc(
    key_facts: List[KeyFact],
    legal_issue: str,
    decision_maker: str = "judge"
) -> NarrativeArc:
    """
    Arrange facts in persuasive order, tailored to audience.
    """

    if decision_maker == "judge":
        # Judges like: Law first, then facts
        opening = f"""
        This case is about whether a partner has a fiduciary duty to
        disclose financial information. The law is clear: they do.
        The facts show defendant violated that duty.
        """

    elif decision_maker == "jury":
        # Juries like: Story, then law
        opening = f"""
        This is a story about betrayal and broken promises.
        Two partners built something together. One broke a promise.
        The law says that's wrong. The facts prove it happened.
        """

    fact_sequence = [f.fact for f in sorted(key_facts, key=lambda x: x.narrative_order)]

    legal_connection = f"""
    These facts show a breach of fiduciary duty because:
    1. Partnership relationship exists (establishes duty)
    2. Duty includes disclosure of material financial information
    3. Defendant refused disclosure (breach)
    4. Plaintiff was harmed (damages)
    """

    conclusion = f"""
    Because defendant breached his fiduciary duty, he is liable.
    The law requires he pay damages.
    """

    return NarrativeArc(
        opening=opening,
        fact_sequence=fact_sequence,
        legal_connection=legal_connection,
        conclusion=conclusion
    )
```

**Step 3: Generate Counter-Narrative**
```python
def construct_opposing_narrative(
    legal_issue: str,
    alternative_interpretation: str,
    client_facts: str
) -> OpposingNarrative:
    """
    What story will the other side tell?
    """

    opening = f"""
    Plaintiff is trying to use boilerplate partnership law to recover
    for his own business miscalculation. This is not a fiduciary duty
    case—it's a case where a partner is unhappy with outcomes.
    """

    key_points = [
        "Plaintiff had opportunity to negotiate better terms",
        "He didn't ask about finances until now",
        "No law requires affirmative disclosure absent request",
        "His damages claim is speculative"
    ]

    legal_argument = f"""
    Disclosure is only required if:
    1. Partner specifically asks (he didn't ask for specifics)
    2. For legitimate partnership purpose (his purpose was to find flaws)
    3. Within reasonable scope (unlimited access isn't required)
    """

    return OpposingNarrative(
        opening=opening,
        key_points=key_points,
        legal_argument=legal_argument
    )
```

### Output

```
OUR NARRATIVE (Persuasive Frame)
═════════════════════════════════════════════════════════

For Judge:
"This case presents a straightforward question of fiduciary duty law.
In a partnership, partners owe each other duties of disclosure and
loyalty. When one partner withholds financial information from another,
that's a breach. Here, the defendant did exactly that."

Key Facts (in order):
1. Partnership formed 2019 (establishes fiduciary relationship)
2. Plaintiff invested $200k (shows stake and good faith)
3. Defendant refused disclosure of company finances (breach)
4. Plaintiff lost business opportunity as result (damages)

How Law Applies:
• Partnership law (statute § 8.01): "Partners have right to access records"
• Case precedent: Smith v. Jones: "Refusal to disclose is per se breach"
• Your facts: Same as Smith (partnership, investment, refusal)
• Conclusion: Breach occurred; damages available

Bottom Line:
"Defendant promised to be honest partners. He broke that promise by
hiding the financials. The law says that's actionable."

THEIR NARRATIVE (What Opponent Will Say)
═════════════════════════════════════════════════════════

"Plaintiff is using partnership law to sue for something that's really
a business dispute. Two partners disagreed about strategy. One wanted
different direction than the other. That's not a legal problem—that's
business."

Their Key Points:
1. Plaintiff could have negotiated better partnership agreement
2. He didn't formally request the information (just complained)
3. Partnership law doesn't require affirmative disclosure
4. His damages are speculative

How They'll Argue Law:
• Statute § 8.01 requires "reasonable request" - he didn't make one
• Smith case is distinguishable (different facts)
• Their interpretation: Disclosure only when specifically asked

OUR VULNERABILITIES (What We Need to Address)
═════════════════════════════════════════════════════════

1. "Didn't formally request" vulnerability
   Risk Level: MODERATE
   Counter: Even without formal request, fiduciary duty exists
   Evidence: Smith case shows refusal without request is breach
   Recommendation: Get testimony that you asked, he refused

2. "Business dispute, not legal issue" vulnerability
   Risk Level: LOW
   Counter: This is exactly what fiduciary duty is designed for
   Evidence: Purpose of statute § 8.01 is to prevent this situation
   Recommendation: Cite policy and similar cases

3. "Damages speculative" vulnerability
   Risk Level: MODERATE
   Counter: Lost opportunity is foreseeable contract damage
   Evidence: Baker Corp. case shows such damages are recoverable
   Recommendation: Document the lost opportunity with specifics

PERSUASION STRATEGY (For Judge)
═════════════════════════════════════════════════════════

1. Lead with the law (judges like precedent)
   - Smith v. Jones clearly established the rule
   - Statute supports plaintiff's interpretation

2. Then apply to facts (shows logic)
   - Facts match precedent
   - Defendant's conduct was breach

3. Address their counter-argument
   - Yes, he didn't make a formal request
   - But fiduciary duty doesn't require formal request
   - Refusal itself is the breach

4. Emphasize policy
   - Partnerships work because of trust and disclosure
   - Defendant violated that trust
   - Law protects plaintiffs in exactly this situation
```

---

## Pattern 6: Context-Aware Recommendation

### What It Does
Recommends legal strategy based on law + client context.

### Process

**Step 1: Understand Actual Client Goals**
```python
def infer_actual_goals(
    stated_goals: str,
    business_context: Dict,
    relationship_context: Dict
) -> ActualGoals:
    """
    What does client REALLY want (vs. what they said)?
    """

    # Client says: "Get full damages"
    # Client's situation: Defendant is only customer
    # Actual goal (inferred): Preserve relationship while getting some recovery

    actual_goals = ActualGoals(
        primary="resolve quickly and move on",
        secondary="preserve business relationship if possible",
        tertiary="avoid lengthy litigation",
        financial_threshold=1.2e6,  # Accept settlement at $1.2M+
        timeline_urgency="high",  # Need resolution by Q2
        risk_tolerance="low"  # Can't afford expensive litigation
    )

    return actual_goals
```

**Step 2: Generate Strategic Options**
```python
def generate_strategic_options(
    legal_case_strength: float,
    client_context: ClientContext
) -> List[StrategicOption]:
    """
    What are the strategic choices?
    """

    options = [
        StrategicOption(
            name="Aggressive Litigation",
            description="File suit immediately, litigate to judgment",
            timeline=24,  # months
            cost=750000,  # dollars
            expected_outcome={
                "success_probability": 0.75,
                "if_win": 2500000,
                "if_lose": 0,
                "expected_value": 1875000
            },
            relationship_impact="destroyed",
            decision_maker_needed="judge",
            risks=[
                "Defendant counter-sues",
                "Litigation is public",
                "Ongoing business relationship destroyed"
            ]
        ),
        StrategicOption(
            name="Settlement Negotiation",
            description="Demand settlement, show we're serious with litigation threat",
            timeline=6,
            cost=75000,
            expected_outcome={
                "success_probability": 0.95,
                "if_settle": 1400000,
                "if_fail_then_litigate": 1875000,
                "expected_value": 1400000
            },
            relationship_impact="preserved but strained",
            decision_maker_needed="opposing_counsel",
            risks=[
                "Defendant demands more than we'll accept",
                "Negotiation breaks down"
            ]
        ),
        StrategicOption(
            name="Mediation",
            description="Use neutral third party to facilitate resolution",
            timeline=3,
            cost=40000,
            expected_outcome={
                "success_probability": 0.70,
                "if_mediate": 1100000,
                "if_fail": 1875000,
                "expected_value": 1100000  # 0.7*1.1M + 0.3*1.875M
            },
            relationship_impact="preserved",
            decision_maker_needed="both_parties",
            risks=[
                "Mediator might split difference unfavorably",
                "Mediation could fail"
            ]
        )
    ]

    return options
```

**Step 3: Score Against Goals**
```python
def score_options_against_goals(
    options: List[StrategicOption],
    actual_goals: ActualGoals
) -> List[ScoredOption]:
    """
    Rank options by how well they meet client's actual goals.
    """

    scored = []

    for option in options:
        score = 0

        # Timeline preference (client wants quick resolution)
        if actual_goals.timeline_urgency == "high":
            timeline_score = 1 - (option.timeline / 24)  # 0-1, higher is faster
            score += timeline_score * 0.25

        # Financial outcome
        # Weight: achieve at least $1.2M with high probability
        financial_score = (
            option.expected_outcome["expected_value"] / 2000000  # max value
        )
        score += min(financial_score, 1.0) * 0.30

        # Relationship preservation
        relationship_values = {
            "destroyed": 0.0,
            "preserved but strained": 0.6,
            "preserved": 1.0
        }
        relationship_score = relationship_values.get(option.relationship_impact, 0.3)
        score += relationship_score * 0.20

        # Budget (can't spend $750k)
        budget_score = 1 - (option.cost / 750000)
        score += max(0, budget_score) * 0.15

        # Risk tolerance (client is risk-averse)
        success_prob = option.expected_outcome["success_probability"]
        risk_score = success_prob if actual_goals.risk_tolerance == "low" else 0.5
        score += risk_score * 0.10

        scored.append(ScoredOption(
            option=option,
            score=score,
            reasoning=[
                f"Timeline score: {timeline_score:.2f} (client wants quick)",
                f"Financial score: {financial_score:.2f} (needs $1.2M+)",
                f"Relationship score: {relationship_score:.2f} (preserves business)",
                f"Budget score: {budget_score:.2f} (limited to $75k legal)",
                f"Risk score: {risk_score:.2f} (wants certainty)"
            ]
        ))

    return sorted(scored, key=lambda x: x.score, reverse=True)
```

### Output

```
STRATEGIC RECOMMENDATION ANALYSIS
═════════════════════════════════════════════════════════

Given: Legal case strength = 75% likely to win
       Client context = quick resolution needed, relationship matters, limited budget

OPTION 1: SETTLEMENT NEGOTIATION ✓✓✓ RECOMMENDED
═════════════════════════════════════════════════════════

Why Recommended:
• Score: 0.82 (best alignment with your actual goals)
• Timeline: 6 months (quick resolution you need)
• Cost: $75k legal fees (fits your budget)
• Outcome: $1.4M-$1.5M expected (meets your threshold)
• Relationship: Preserved but strained (can continue business)

How It Works:
1. Prepare litigation materials (show we're serious)
2. Open settlement discussions at $1.5M
3. Negotiate range $1.2M-$1.5M
4. 95% likely to reach agreement within this range

If Successful: You get $1.4M, relationship continues, move on in 6 months
If Fails: You proceed to litigation (fallback option)

OPTION 2: LITIGATION
═════════════════════════════════════════════════════════

Score: 0.58 (lower alignment with goals)
Why Lower:
• Timeline: 24 months (too long, you need resolution by Q2)
• Cost: $750k (exceeds your budget 10x)
• Relationship: Destroyed (important to you)
• Probability: 75% success (you want certainty)

When to Choose This:
• If settlement negotiations fail
• If defendant refuses reasonable offer
• If you decide relationship is worth destroying for max recovery

Expected Outcome: $2.5M (but high risk, high cost, takes 2 years)

OPTION 3: MEDIATION
═════════════════════════════════════════════════════════

Score: 0.72 (good option, backup if settlement talks fail)
Why Lower Than Settlement:
• Success probability: 70% vs. 95% for settlement
• Outcome: $1.1M (might be below your threshold)
• Mediator might split difference in ways you don't like

When to Choose:
• If direct settlement talks stall
• If you want genuinely neutral third party
• If relationship preservation is priority #1

TRADE-OFFS OF RECOMMENDED OPTION (Settlement)
═════════════════════════════════════════════════════════

What You're Giving Up:
├─ Potential to win full $2.5M (75% chance)
│  → Cost: Give up ~$1M in potential recovery
│
└─ Vindication (settlement doesn't prove you were right)
   → Cost: No public acknowledgment of defendant's wrongdoing

What You're Getting:
├─ Certainty (95% likely to reach settlement)
│  → Benefit: Know the outcome, can plan
│
├─ Speed (6 months vs. 24 months)
│  → Benefit: Quick closure, move on with your life
│
├─ Relationship (preserved but strained)
│  → Benefit: Can work with defendant again if needed
│
├─ Budget (fits $75k legal budget)
│  → Benefit: Don't deplete your capital on litigation
│
└─ Risk (no chance of losing everything)
   → Benefit: Worst case is mediation fails, then you litigate

Net Assessment: For your situation, these benefits are worth ~$1M.

DECISION MATRIX
═════════════════════════════════════════════════════════

                Settlement    Mediation    Litigation
Timeline          6 months      3 months    24 months
Cost              $75k          $40k        $750k
Relationship      Preserved     Preserved   Destroyed
Expected $        $1.4M         $1.1M       $1.875M
Success Prob      95%           70%         75%
Certainty         Very High     Moderate    Moderate
Budget Fit        ✓             ✓✓          ✗

NEXT STEPS
═════════════════════════════════════════════════════════

1. Confirm with client:
   ☐ Is $1.4M acceptable? (vs. risk of $0 in litigation)
   ☐ Is preserving relationship important?
   ☐ Can you wait 6 months for resolution?
   ☐ Do you have $75k budget for legal?

2. If yes to all:
   ☐ Gather litigation materials
   ☐ Send settlement demand at $1.5M
   ☐ Open negotiations

3. Target settlement: $1.2M-$1.5M
   ☐ Below $1.2M: recommend mediation
   ☐ Above $1.5M: recommend accepting

4. If settlement fails:
   ☐ Proceed to litigation or mediation
   ☐ This fallback is your safety net
```

---

## Summary: Implementation Priority

**Phase 1 (Easiest, Highest ROI):**
1. Problem Formulation (Pattern 1)
2. Confidence Scoring (Pattern 4)

**Phase 2 (Medium Complexity):**
3. Schema Organization (Pattern 3)
4. Multi-Hypothesis Generation (Pattern 2)

**Phase 3 (Higher Complexity):**
5. Narrative Construction (Pattern 5)
6. Context-Aware Recommendation (Pattern 6)

Start with patterns 1 and 4. These two alone will dramatically improve your system.

---

## Technical Integration Points

Where to add these in your current architecture:

```
Current Flow:
Query → Embed → Search → Rank → Generate Answer

Enhanced Flow:
Query
  ↓
[NEW] Problem Formulation Engine → Identify actual legal issue
  ↓
Search (now issue-focused, not query-focused)
  ↓
[NEW] Classification Engine → Categorize results by schema
  ↓
[NEW] Multi-Hypothesis Generator → Generate alternatives
  ↓
Rank
  ↓
[NEW] Confidence Scorer → Score confidence in conclusions
  ↓
[NEW] Narrative Constructor → Help build persuasive story
  ↓
[NEW] Context Analyzer → Integrate client context
  ↓
Generate Answer (with all the above context)
```

Each new layer adds cognitive support without replacing the existing architecture.
