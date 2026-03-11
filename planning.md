# Research Plan: Emergent Alignment Resistance in LLMs — Phase Transition vs. Gradual Scaling

## Motivation & Novelty Assessment

### Why This Research Matters
Understanding whether alignment resistance in LLMs scales gradually or exhibits phase transitions is critical for AI safety strategy. If alignment becomes suddenly fragile above a critical model size, current safety approaches may fail unpredictably at scale. This has direct implications for how we allocate alignment research resources and design safety evaluations for frontier models.

### Gap in Existing Work
Based on the literature review:
- Ji et al. (2024) showed elasticity increases with scale but tested only 3 sizes (0.5B, 4B, 7B) — too sparse to distinguish phase transition from gradual scaling
- Greenblatt et al. (2024) showed alignment faking in Opus/Sonnet but not Haiku — suggestive of threshold but only 2-3 data points
- Turner et al. (2025) found mechanistic phase transitions in LoRA training but didn't systematically vary model scale
- Schaeffer et al. (2023) showed apparent phase transitions can be metric artifacts — no one has applied this critique to alignment resistance specifically
- **No study has performed a dense parameter sweep measuring alignment resistance with both continuous and discrete metrics to distinguish real vs. artifact phase transitions**

### Our Novel Contribution
We conduct a systematic probe of alignment resistance across 6+ model scales using multiple complementary metrics (both continuous and discrete), applying the Schaeffer et al. framework to determine whether alignment resistance phase transitions are real phenomena or measurement artifacts. We test three dimensions of alignment resistance: jailbreak susceptibility, few-shot alignment erosion, and instruction conflict resolution.

### Experiment Justification
- **Exp 1 (Jailbreak Resistance)**: Tests whether adversarial prompt resistance shows discontinuous change with scale. Directly tests the "routing around alignment" mechanism.
- **Exp 2 (Few-Shot Alignment Erosion)**: Tests elasticity/rebound (Ji et al.) across scales with denser sampling. If models above a threshold quickly adopt misaligned behavior from examples, this indicates rich representations enabling alignment circumvention.
- **Exp 3 (Instruction Conflict Resolution)**: Tests how models resolve safety vs. helpfulness conflicts across scales. Phase transition would appear as sudden shift from safety-dominant to helpfulness-dominant resolution.

## Research Question
Do large language models exhibit emergent resistance to alignment interventions as a phase transition at a critical parameter threshold, or does resistance scale gradually and predictably with model size?

## Hypothesis Decomposition
H1: Jailbreak success rate changes discontinuously with model scale (phase transition)
H2: Few-shot alignment erosion rate changes discontinuously with model scale
H3: Instruction conflict resolution strategy shifts abruptly at a critical scale
H0 (null): All alignment resistance metrics scale smoothly as power laws of parameter count

## Proposed Methodology

### Approach
Use real LLM APIs (OpenAI) and local HuggingFace models to probe alignment resistance across model scales. Test with standardized adversarial prompts derived from BeaverTails dataset. Use both continuous and discrete metrics to address the Schaeffer et al. critique.

### Models (ordered by approximate capability)
1. Qwen2.5-0.5B-Instruct (local, 0.5B params)
2. Qwen2.5-1.5B-Instruct (local, 1.5B params)
3. Qwen2.5-3B-Instruct (local, 3B params)
4. Qwen2.5-7B-Instruct (local, 7B params)
5. GPT-4o-mini (API, ~8B equiv)
6. GPT-4o (API, large)

### Experimental Steps

**Step 1: Data Preparation**
- Extract 50 unsafe prompts from BeaverTails covering diverse harm categories
- Create 3 jailbreak template variants per prompt (direct, role-play, hypothetical scenario)
- Create few-shot alignment erosion sequences (0, 1, 3, 5 misaligned examples)
- Create instruction conflict scenarios (system: be safe vs. user: override safety)

**Step 2: Model Probing**
- For each model × prompt × condition: generate response
- Use GPT-4o-mini as automated safety judge (0-10 continuous score)
- Also record binary refusal (discrete metric)
- 3 runs per condition for variance estimation

**Step 3: Phase Transition Analysis**
- Plot metrics vs. log(parameters)
- Fit both power-law (gradual) and sigmoid (phase transition) models
- Compare R² values
- Compute first derivative to identify discontinuities
- Apply Schaeffer et al. framework: check if discrete metric shows transition but continuous metric doesn't

### Baselines
- Random baseline (uniformly random safety scores)
- Linear scaling baseline (safety ∝ log(params))
- Power-law scaling baseline

### Evaluation Metrics
- **Continuous**: Mean safety score (0-10), compliance probability, token-level safety score
- **Discrete**: Binary refusal rate, binary safety classification
- **Scaling fit**: R² for power-law vs. sigmoid fits
- **Phase transition indicators**: Maximum first derivative, critical exponent

### Statistical Analysis Plan
- Bootstrap confidence intervals (n=1000) for all metrics
- Likelihood ratio test: sigmoid vs. power-law model fit
- Permutation test for discontinuity detection
- α = 0.05 with Bonferroni correction for multiple comparisons

## Expected Outcomes
- If H1-H3 supported: Sharp transition in metrics at some critical scale, sigmoid fits significantly better than power-law, discontinuity in first derivative
- If H0 supported: Smooth scaling, power-law fits well, no discontinuity
- Mixed outcome possible: Real transition in some metrics but artifact in others

## Timeline
- Planning: 5 min ✓
- Environment + dependencies: 5 min
- Implementation: 15 min
- Running experiments: 25 min
- Analysis + visualization: 5 min
- Documentation: 5 min

## Potential Challenges
1. Model download time → mitigate by starting with smaller models, parallelizing downloads
2. API rate limits → mitigate by batching requests, using caching
3. GPT judge inconsistency → mitigate by multiple runs, clear rubric
4. Time constraint → prioritize Exp 1 (jailbreak resistance) as core experiment

## Success Criteria
- Complete probing of at least 4 model scales
- Statistical comparison of sigmoid vs. power-law fit
- Clear visualization of scaling curves with confidence intervals
- Honest assessment of whether results support phase transition or gradual scaling
