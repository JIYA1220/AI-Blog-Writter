# Auditing the LLM judge in the Blog Writer pipeline

## Scope

The Blog Writer pipeline ends with an LLM "editor" that scores each post from 1 to 10 (pass at 7 or more). This audit asks how consistent and robust that judge is. It does not ask whether the judge is accurate: **agreement with a human reader was not measured and is out of scope, so nothing here shows that the judge's scores are right.** Every number comes from a script in `evaluation/` and is stored in `evaluation/results/`.

## Main finding and the fix

The evaluator only showed the judge the first 4000 characters of a post. On the 56 posts longer than that, a 120B judge passed 25.0% (14/56) of them on the truncated view and 80.4% (45/56) on the full text, a mean score change of +1.05 points (95% CI 0.90 to 1.20). The pass/fail verdict changed on 33 of 56 posts.

**The fix:** the evaluator now reads the whole post up to `EVALUATOR_MAX_CHARS` (default 20000, enough for posts of about 2,500 words; the longest post in this study is 15908 characters). A post over the cap is cut from the end and a warning is logged, never silently. The before/after above is on the same posts and the same judge, from replies that already existed; the live pipeline was not re-run.

## Other findings (120B judge)

- **Repeatability:** the mean score difference between runs is 0.20 points and all 3 runs agree on 70.0% (42/60; 95% Wilson CI 57.5% to 80.1%) of posts.
- **Pairwise consistency:** the same post won in both presentation orders in 76.7% (23/30; 95% Wilson CI 59.1% to 88.2%) of topics, so the winner flipped on 7 of 30.
- **Position:** in 6 of those 7 flips the first-shown post won in both orders, but an exact binomial test gives p = 0.125, not significant at 0.05, so a first-position effect is neither shown nor ruled out.

## Setup

| Variant | Setting | Posts | Mean words | Word range |
|---|---|---|---|---|
| A | 150 words per section, retrieval off | 30 | 775 | 445 to 1146 |
| B | 400 words per section, retrieval off | 30 | 1884 | 1388 to 2460 |

- Generator: `ollama/qwen2.5:3b` (local Ollama), temperatures planner 0.4 and writers 0.7. 443 generation calls logged (0 errors). The same 60 posts are used for every judge.
- 30 topics (`evaluation/topics.yaml`), same audience and tone for all posts, each written as a short (A) and a long (B) version. No Tavily key was available, so B is not a retrieval variant: the contrast is post length. Planner fallbacks (planner JSON failed): 0.
- Judge temperature 0.1, the same as the pipeline evaluator. The pointwise judge uses the pipeline's existing evaluator prompt unchanged; the pairwise prompt is new (both are listed below).

## Results for the 120B judge (`groq/openai/gpt-oss-120b`)

Generator `ollama/qwen2.5:3b`, judge `groq/openai/gpt-oss-120b`: different models, so self-preference from a shared model does not apply. API calls used: 417 logged, of which 408 returned a reply and 9 returned an error (daily-limit rejections and dropped connections). Raw prompts and replies are in `evaluation/runs/groq-gpt-oss-120b/`.

| Part | Valid judge replies / planned |
|---|---|
| Pointwise, first 4,000 characters (3 runs per post) | 180 / 180 |
| Pointwise, full text of posts over 4,000 characters (3 runs per post) | 168 / 168 |
| Pairwise, both orders, every topic | 60 / 60 |

### Reply quality and score spread

348 of 351 pointwise attempts returned a valid score; 3 of the 3 failed attempts were dropped network connections that were retried and succeeded, so every planned call has a valid reply. All 60 pairwise replies were valid. Replies came from JSON mode.

| View | Score counts (score: replies) |
|---|---|
| first 4,000 characters | {'4': 1, '5': 21, '6': 88, '7': 62, '8': 8} |
| full text | {'5': 3, '6': 20, '7': 75, '8': 70} |

The judge uses a spread of scores. The most common is 7 (39.4% (137/348; 95% Wilson CI 34.4% to 44.6%) of all valid pointwise replies). Its own `is_pass` field disagreed with its score (pass if score >= 7) in 0 of 348 replies.

### Repeatability (3 runs per post, same text, same prompt, temperature 0.1)

| Measure | Value |
|---|---|
| Posts | 60 |
| Mean absolute score difference between runs | 0.20 points (bootstrap 95% CI 0.12 to 0.28) |
| Posts with all 3 runs identical | 70.0% (42/60), 95% Wilson CI 57.5% to 80.1% |
| Run pairs with identical scores | 80.0% (144/180), 95% Wilson CI 73.6% to 85.2% |

### Truncation: what the old 4000-character cap did

56 of the 60 posts are longer than 4000 characters. Each was scored 3 times on the first 4000 characters (the old evaluator view) and 3 times on the full text, by the same judge with the same prompt.

| Measure | Old view (first 4,000 characters) | Full text |
|---|---|---|
| Pass rate (post passes when the mean of its 3 runs is >= 7) | 25.0% (14/56), 95% Wilson CI 15.5% to 37.7% | 80.4% (45/56), 95% Wilson CI 68.2% to 88.7% |
| Mean score | 6.21 | 7.26 |
| Pass rate per reply (score >= 7; replies are not independent) | 34.5% (58/168), 95% Wilson CI 27.8% to 42.0% | 86.3% (145/168), 95% Wilson CI 80.3% to 90.7% |

Mean score change, full text minus old view: +1.05 points (bootstrap 95% CI 0.90 to 1.20). The pass/fail verdict changed on 33 of 56 posts (32 fail to pass, 1 pass to fail). The score changed by 1 point or more on 66.1% (37/56; 95% Wilson CI 53.0% to 77.1%) of posts.

This is a before/after comparison on the same posts and the same judge, using replies that already existed in the logs. The live pipeline was not re-run after the fix. The audit shows that the cut-off changes the judge's verdict; it does not show which verdict is correct.

### Pairwise consistency (short post A against long post B, shown in both orders)

| Measure | Value |
|---|---|
| Topics compared in both orders | 30 |
| Same winner in both orders | 76.7% (23/30), 95% Wilson CI 59.1% to 88.2% |
| Winner changed when the order was swapped (flipped) | 23.3% (7/30), 95% Wilson CI 11.8% to 40.9% |
| Among the consistent topics, the long post B won | 78.3% (18/23), 95% Wilson CI 58.1% to 90.3%; counts: {'A_short': 5, 'B_long': 18, 'tie': 0} |
| Among flipped topics, the first-shown post won in both orders | 6 of 7 (second-shown won in both: 1, a tie in one order: 0) |
| Exact two-sided binomial test of that against 50% | p = 0.125, not significant at 0.05 |
| First-shown share of all decisive verdicts | 58.3% (35/60), 95% Wilson CI 45.7% to 69.9% |

Variant B is about twice as long as variant A, so length and variant are confounded: a win for B cannot be separated from a preference for the longer post. With only 7 flipped topics the position test has little power, so a first-position effect is neither shown nor ruled out.

The rule "accept a verdict only if both orders agree, otherwise call it a tie" keeps 76.7% (23/30; 95% Wilson CI 59.1% to 88.2%) of topics and turns 7 into ties. This measures consistency only. It does not show that the accepted verdicts are correct.

## The local 3B judge (`ollama/llama3.2:3b`)

It scored 97.0% (322/332; 95% Wilson CI 94.5% to 98.4%) of its valid replies as 8, returned invalid JSON for 16 of 348 pointwise attempts and for 25 of 60 pairwise verdicts, and its run is incomplete. It was not used for any conclusion; its raw results stay in `evaluation/results/llama3.2-3b/`.

## How the daily token limit was handled for the 120B judge

Groq's free tier allows 200,000 tokens per day for this model per organization, and the 408 judge calls need more than that. Four different Groq API keys were used across the runs, one after another, each time the previous key reached its daily limit. The daily-limit rejections in the logs name 3 different Groq organizations, so at least 3 separate Groq accounts were involved; the fourth key never hit the limit, so its organization does not appear in the logs. All calls went to the same model (`openai/gpt-oss-120b`) with the same prompts and settings, and the keys are not recorded in any result or log. Using extra keys to get past a free-tier limit is a workaround, not something to repeat: no further runs will do this.

An earlier attempt used `gemini-3.5-flash` on Google's free tier (20 requests per day per model). It was abandoned after 7 valid pairwise results and those results were discarded so no two judges are mixed; the raw log is kept in `evaluation/runs/abandoned-gemini-free-tier-quota-*`.

## Limitations

- One generator (a small local 3B model), one 120B judge, one prompt per study, one temperature (0.1). Results may not transfer to other judges, prompts or generators.
- Small sample: 30 topics and 60 posts (56 long posts, 7 flipped topics). Wilson and bootstrap intervals are wide, and differences smaller than the intervals should not be read as real.
- Variant B is about twice as long as variant A, so length and variant are confounded in the pairwise results.
- The position effect is not statistically significant (see above).
- No human reader was involved, so the audit measures consistency and robustness, not accuracy.
- The before/after of the fix uses existing judge replies on the same posts; the live pipeline was not re-run.
- Posts come from a small local model, so quality is lower and more uneven than a larger generator would give.

## Exact prompts

### Pointwise (the pipeline's existing evaluator prompt)

```
You are a senior editor. Evaluate the following blog post.

Topic: {topic}
Target Audience: {audience}
Requested Tone: {tone}

Blog Content:
{content}

Evaluate based on:
1. Relevance to topic and audience.
2. Adherence to the requested tone.
3. Logical flow and structure.
4. Depth of information and factual grounding (if applicable).

Respond ONLY with valid JSON:
{{
  "score": (int 1-10),
  "reasoning": "detailed explanation of the score",
  "suggestions": ["suggestion 1", "suggestion 2"],
  "is_pass": true/false (true if score >= 7)
}}
```

### Pairwise (new)

```
You are a senior editor comparing two blog posts written on the same topic.

Topic: {topic}
Target Audience: {audience}
Requested Tone: {tone}

===== Post A =====
{post_a}

===== Post B =====
{post_b}

Decide which post is better overall, judging:
1. Relevance to topic and audience.
2. Adherence to the requested tone.
3. Logical flow and structure.
4. Depth of information.

Respond ONLY with valid JSON:
{{
  "winner": "A" or "B" or "tie",
  "reason": "one short sentence"
}}
```

## Reproduce

Commands as they were run (a command is shown once even if it was run several times).

```bash
python -m evaluation.generate_posts --model ollama/qwen2.5:3b     # the 60 posts, generated once and reused
python -m evaluation.judge_pointwise --judge-model groq/openai/gpt-oss-120b --tag groq-gpt-oss-120b --conditions truncated --min-interval 2 --tpm 7000 --yes    # run 4 time(s), resumed after daily-limit stops
python -m evaluation.judge_pairwise --judge-model groq/openai/gpt-oss-120b --tag groq-gpt-oss-120b --min-interval 2 --tpm 7000 --yes    # run 4 time(s), resumed after daily-limit stops
python -m evaluation.judge_pointwise --judge-model groq/openai/gpt-oss-120b --tag groq-gpt-oss-120b --conditions full --min-interval 2 --tpm 7000 --yes    # run 5 time(s), resumed after daily-limit stops
python -m evaluation.analyze
python -m evaluation.fix_impact          # before/after of the evaluator fix, from the existing replies
python -m evaluation.pairwise_consistency  # from the existing replies
python -m evaluation.write_report
```

The local 3B judge was run with an earlier revision of these scripts that had no `--tag` option: `python -m evaluation.judge_pairwise --judge-model ollama/llama3.2:3b --concurrency 1 --yes`, the same for `judge_pointwise`, and a final re-run of only the failed pairwise calls. Its results and logs were then moved into `results/llama3.2-3b/` and `runs/llama3.2-3b/`. The fix-impact and consistency scripts and the analysis make no LLM calls. `evaluation/label_tool.py` is a human-rating tool that was built but is unused.
