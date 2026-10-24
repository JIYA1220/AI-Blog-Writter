# Auditing the LLM judge in the Blog Writer pipeline

A short summary. The full tables, prompts and limitations are in [`REPORT.md`](REPORT.md). Every number comes from a script in `evaluation/` and is stored in `evaluation/results/`.

The Blog Writer pipeline ends with an LLM "editor" that scores each finished post from 1 to 10, with 7 or more counting as a pass. This audit checked how consistent and robust that judge is, using 60 generated posts (30 topics, each written as a short and a long version) and a 120B judge. It did not check whether the judge is accurate: agreement with a human reader was not measured and is out of scope.

## What the audit found

**1. The evaluator's 4000-character cap changed the verdict on most long posts, so it was fixed.** The evaluator only showed the judge the first 4000 characters. On the 56 posts longer than that, the pass rate was 25.0% (14/56; 95% Wilson CI 15.5% to 37.7%) on the truncated view and 80.4% (45/56; 95% Wilson CI 68.2% to 88.7%) on the full text. The mean score rose by 1.05 points (95% CI 0.90 to 1.20) and the pass/fail verdict changed on 33 of 56 posts. The evaluator now reads the whole post up to `EVALUATOR_MAX_CHARS` (default 20000) and logs a warning if it ever has to cut one. This is a before/after on the same posts and the same judge, taken from replies that already existed; the live pipeline was not re-run after the fix.

**2. Pairwise verdicts were fairly but not fully consistent.** When a short and a long post were compared in both orders, the same post won both times on 76.7% (23/30; 95% Wilson CI 59.1% to 88.2%) of topics. In 6 of the 7 flipped topics the first-shown post won in both orders, but an exact binomial test gives p = 0.125, which is not significant at 0.05, so a first-position effect is neither shown nor ruled out. Among the consistent topics the long post won 78.3% (18/23; 95% Wilson CI 58.1% to 90.3%); because the long post is about twice as long, that cannot be separated from a preference for length.

**3. Scores were mostly repeatable.** Across 3 runs on the same text the mean score difference was 0.20 points (95% CI 0.12 to 0.28), and all 3 runs agreed on 70.0% (42/60; 95% Wilson CI 57.5% to 80.1%) of posts.

## What this does not show

- Nothing here shows the judge is accurate. No human read the posts, so there is no comparison with a person.
- The pairwise results measure consistency only. Accepting a verdict only when both orders agree keeps 76.7% (23/30; 95% Wilson CI 59.1% to 88.2%) of topics, but that does not make those verdicts correct.
- A local 3B judge was also tried. It gave almost every post a score of 8 and often returned invalid JSON, so its run is incomplete and it was not used for any conclusion.

## Limitations

- One generator (a small local 3B model), one 120B judge, one prompt per study, one temperature (0.1).
- Small sample: 60 posts, 56 long posts, 7 flipped topics. The confidence intervals are wide.
- The long and short versions differ in length, so length and variant are confounded in the pairwise results.
- The position effect is not statistically significant.

## How the daily token limit was handled for the 120B judge

Groq's free tier allows 200,000 tokens per day for this model per organization, and the 408 judge calls need more than that. Four different Groq API keys were used across the runs, one after another, each time the previous key reached its daily limit. The daily-limit rejections in the logs name 3 different Groq organizations, so at least 3 separate Groq accounts were involved; the fourth key never hit the limit, so its organization does not appear in the logs. All calls went to the same model (`openai/gpt-oss-120b`) with the same prompts and settings, and the keys are not recorded in any result or log. Using extra keys to get past a free-tier limit is a workaround, not something to repeat: no further runs will do this.

An earlier attempt used `gemini-3.5-flash` on Google's free tier (20 requests per day per model). It was abandoned after 7 valid pairwise results and those results were discarded so no two judges are mixed; the raw log is kept in `evaluation/runs/abandoned-gemini-free-tier-quota-*`.

## Where things are

- `REPORT.md`: full tables, exact prompts, and the commands that were run.
- `results/`: all computed numbers (`fix_impact.json`, and per judge `analysis.json` and `pairwise_consistency.json`).
- `runs/`: every prompt and raw reply, verbatim.
- `posts/`: the 60 generated posts. `label_tool.py` is a human-rating tool that was built but is unused.
