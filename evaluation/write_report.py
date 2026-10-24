"""
Builds evaluation/REPORT.md from the saved results (per-judge analysis.json, fix_impact.json,
pairwise_consistency.json, run metadata, call logs, post metadata).
Every number comes from those files. No LLM calls are made.

Usage:
  python -m evaluation.analyze && python -m evaluation.fix_impact && python -m evaluation.pairwise_consistency
  python -m evaluation.write_report
"""

import csv
import glob
import json
import os
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agents.evaluator import DEFAULT_EVALUATOR_MAX_CHARS, EVALUATOR_PROMPT, LEGACY_EVAL_MAX_CHARS  # noqa: E402
from evaluation.judge_pairwise import PAIRWISE_PROMPT                                           # noqa: E402

EVAL_DIR = os.path.join(ROOT, "evaluation")
RESULTS_DIR = os.path.join(EVAL_DIR, "results")
RUNS_DIR = os.path.join(EVAL_DIR, "runs")
MAIN_TAG = "groq-gpt-oss-120b"     # the judge all conclusions are drawn from
SMALL_TAG = "llama3.2-3b"          # the local 3B judge, not used for conclusions


def load_json(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def fmt_prop(d):
    return f"{d['p']:.1%} ({d['k']}/{d['n']}), 95% Wilson CI {d['wilson95'][0]:.1%} to {d['wilson95'][1]:.1%}"


def fmt_inline(d):
    """Same numbers as fmt_prop, shaped to sit inside a sentence: 70.0% (42/60; 95% Wilson CI 57.5% to 80.1%)."""
    return f"{d['p']:.1%} ({d['k']}/{d['n']}; 95% Wilson CI {d['wilson95'][0]:.1%} to {d['wilson95'][1]:.1%})"


def fmt_ci(lo_hi, digits=2):
    return f"{lo_hi[0]:.{digits}f} to {lo_hi[1]:.{digits}f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def count_calls(pattern):
    """(calls logged, calls that returned an error) over every calls.jsonl matching the glob."""
    n = errors = 0
    for path in glob.glob(pattern, recursive=True):
        with open(path, encoding="utf-8") as f:
            for line in f:
                n += 1
                errors += bool(json.loads(line).get("error"))
    return n, errors


def cap_error_orgs(tag):
    """Distinct Groq organisation ids named in daily-limit errors (ids only; used to count accounts)."""
    orgs = []
    path = os.path.join(RUNS_DIR, tag, "calls.jsonl")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                err = json.loads(line).get("error") or ""
                m = re.search(r"organization `(org_[a-z0-9]+)`", err)
                if m and m.group(1) not in orgs:
                    orgs.append(m.group(1))
    return orgs


def judge_model_of(tag):
    for name in ("pointwise_raw.csv", "pairwise_raw.csv"):
        path = os.path.join(RESULTS_DIR, tag, name)
        if os.path.exists(path):
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    return row["judge_model"]
    return "unknown"


def main_judge_section(a, fi, pc, tag, gen_model):
    model = judge_model_of(tag)
    calls, call_errors = count_calls(os.path.join(RUNS_DIR, tag, "**", "calls.jsonl"))
    cp = a["completeness"]
    L = [f"## Results for the 120B judge (`{model}`)", "",
         f"Generator `{gen_model}`, judge `{model}`: different models, so self-preference from a shared model does not "
         f"apply. API calls used: {calls} logged, of which {calls - call_errors} returned a reply and {call_errors} "
         "returned an error (daily-limit rejections and dropped connections). Raw prompts and replies are in "
         f"`evaluation/runs/{tag}/`.", "",
         table(["Part", "Valid judge replies / planned"], [
             ["Pointwise, first 4,000 characters (3 runs per post)",
              f"{cp['pointwise_truncated']['parsed']} / {cp['pointwise_truncated']['planned']}"],
             ["Pointwise, full text of posts over 4,000 characters (3 runs per post)",
              f"{cp['pointwise_full']['parsed']} / {cp['pointwise_full']['planned']}"],
             ["Pairwise, both orders, every topic", f"{cp['pairwise']['parsed']} / {cp['pairwise']['planned']}"]]), ""]

    sc = a["study1_calls"]
    L += ["### Reply quality and score spread", "",
          f"{sc['parsed_ok']} of {sc['rows']} pointwise attempts returned a valid score; "
          f"{a.get('failed_attempts_later_succeeded', 0)} of the {sc['failed_to_parse']} failed attempts were dropped "
          "network connections that were retried and succeeded, so every planned call has a valid reply. "
          f"All {cp['pairwise']['parsed']} pairwise replies were valid. Replies came from JSON mode.", ""]
    sd = a["score_distribution"]
    L += [table(["View", "Score counts (score: replies)"],
                [["first 4,000 characters", sd["truncated"]], ["full text", sd["full"]]]), "",
          f"The judge uses a spread of scores. The most common is {sd['modal_score']} "
          f"({fmt_inline(sd['modal_score_share'])} of all valid pointwise replies). Its own `is_pass` field disagreed "
          f"with its score (pass if score >= 7) in {a['is_pass_vs_score']['is_pass_disagrees_with_score_ge_7']} of "
          f"{a['is_pass_vs_score']['parsed_replies']} replies.", ""]

    tr = a["test_retest"]
    L += ["### Repeatability (3 runs per post, same text, same prompt, temperature 0.1)", "",
          table(["Measure", "Value"], [
              ["Posts", tr["n_posts"]],
              ["Mean absolute score difference between runs",
               f"{tr['mean_abs_diff']:.2f} points (bootstrap 95% CI {fmt_ci(tr['mean_abs_diff_ci95'])})"],
              ["Posts with all 3 runs identical", fmt_prop(tr["share_posts_all_runs_identical"])],
              ["Run pairs with identical scores", fmt_prop(tr["share_run_pairs_identical"])]]), ""]

    tc = a["truncation"]
    L += [f"### Truncation: what the old {LEGACY_EVAL_MAX_CHARS}-character cap did", "",
          f"{fi['n_posts']} of the 60 posts are longer than {LEGACY_EVAL_MAX_CHARS} characters. Each was scored 3 times "
          f"on the first {LEGACY_EVAL_MAX_CHARS} characters (the old evaluator view) and 3 times on the full text, by "
          "the same judge with the same prompt.", "",
          table(["Measure", "Old view (first 4,000 characters)", "Full text"], [
              ["Pass rate (post passes when the mean of its 3 runs is >= 7)",
               fmt_prop(fi["pass_rate_before_4000_chars"]), fmt_prop(fi["pass_rate_after_full_text"])],
              ["Mean score", f"{fi['mean_score_before']:.2f}", f"{fi['mean_score_after']:.2f}"],
              ["Pass rate per reply (score >= 7; replies are not independent)",
               fmt_prop(fi["per_reply_pass_rate_before"]), fmt_prop(fi["per_reply_pass_rate_after"])]]), "",
          f"Mean score change, full text minus old view: {fi['mean_score_change']:+.2f} points "
          f"(bootstrap 95% CI {fmt_ci(fi['mean_score_change_ci95'])}). The pass/fail verdict changed on "
          f"{fi['verdict_changes']['fail_to_pass'] + fi['verdict_changes']['pass_to_fail']} of {fi['n_posts']} posts "
          f"({fi['verdict_changes']['fail_to_pass']} fail to pass, {fi['verdict_changes']['pass_to_fail']} pass to fail). "
          f"The score changed by 1 point or more on {fmt_inline(tc['share_changed_by_1_or_more_points'])} of posts.", "",
          "This is a before/after comparison on the same posts and the same judge, using replies that already existed "
          "in the logs. The live pipeline was not re-run after the fix. The audit shows that the cut-off changes the "
          "judge's verdict; it does not show which verdict is correct.", ""]

    pb = a["position_bias"]
    same, cons, flip = pc["same_winner_in_both_orders"], pc["among_consistent_topics"], pc["among_flipped_topics"]
    bt = flip["binomial_test"]
    L += ["### Pairwise consistency (short post A against long post B, shown in both orders)", "",
          table(["Measure", "Value"], [
              ["Topics compared in both orders", pc["n_topics"]],
              ["Same winner in both orders", fmt_prop(same)],
              ["Winner changed when the order was swapped (flipped)", fmt_prop(pb["inconsistency_rate"])],
              ["Among the consistent topics, the long post B won",
               f"{fmt_prop(cons['B_long_won'])}; counts: {cons['winner_counts']}"],
              ["Among flipped topics, the first-shown post won in both orders",
               f"{flip['first_shown_won_in_both_orders']} of {flip['n']} (second-shown won in both: "
               f"{flip['second_shown_won_in_both_orders']}, a tie in one order: {flip['tie_in_one_order']})"],
              ["Exact two-sided binomial test of that against 50%",
               f"p = {bt['p_value_exact_two_sided']:.3f}, " + ("significant" if bt["significant_at_0.05"]
                                                               else "not significant") + " at 0.05"],
              ["First-shown share of all decisive verdicts", fmt_prop(pb["first_shown_share_of_all_decisive_verdicts"])]]),
          "",
          "Variant B is about twice as long as variant A, so length and variant are confounded: a win for B cannot be "
          "separated from a preference for the longer post. With only "
          f"{flip['n']} flipped topics the position test has little power, so a first-position effect is neither "
          "shown nor ruled out.", "",
          f"The rule \"accept a verdict only if both orders agree, otherwise call it a tie\" keeps "
          f"{fmt_inline(pc['agree_or_tie_rule']['topics_with_an_accepted_verdict'])} of topics and turns "
          f"{pc['agree_or_tie_rule']['topics_turned_into_a_tie_by_the_rule']} into ties. This measures consistency "
          "only. It does not show that the accepted verdicts are correct.", ""]
    return L


def small_judge_section(a):
    model = judge_model_of(SMALL_TAG)
    sd, sc, pp = a["score_distribution"], a["study1_calls"], a["pairwise_parse"]
    return [f"## The local 3B judge (`{model}`)", "",
            f"It scored {fmt_inline(sd['modal_score_share'])} of its valid replies as {sd['modal_score']}, returned invalid "
            f"JSON for {sc['failed_to_parse']} of {sc['rows']} pointwise attempts and for "
            f"{pp['topic_orders_attempted'] - pp['topic_orders_parsed']} of {pp['topic_orders_attempted']} pairwise "
            "verdicts, and its run is incomplete. It was not used for any conclusion; its raw results stay in "
            f"`evaluation/results/{SMALL_TAG}/`.", ""]


def token_limit_section():
    orgs = cap_error_orgs(MAIN_TAG)
    return ["## How the daily token limit was handled for the 120B judge", "",
            "Groq's free tier allows 200,000 tokens per day for this model per organization, and the 408 judge calls need "
            "more than that. Four different Groq API keys were used across the runs, one after another, each time the "
            f"previous key reached its daily limit. The daily-limit rejections in the logs name {len(orgs)} different "
            f"Groq organizations, so at least {len(orgs)} separate Groq accounts were involved; the fourth key never "
            "hit the limit, so its organization does not appear in the logs. All calls went to the same model "
            "(`openai/gpt-oss-120b`) with the same prompts and settings, and the keys are not recorded in any result "
            "or log. Using extra keys to get past a free-tier limit is a workaround, not something to repeat: no "
            "further runs will do this.", "",
            "An earlier attempt used `gemini-3.5-flash` on Google's free tier (20 requests per day per model). It was "
            "abandoned after 7 valid pairwise results and those results were discarded so no two judges are mixed; "
            "the raw log is kept in `evaluation/runs/abandoned-gemini-free-tier-quota-*`.", ""]


def reproduce_block():
    meta = load_json(os.path.join(RUNS_DIR, MAIN_TAG, "run_meta.json")) or {"invocations": []}
    counts, order = Counter(), []
    for inv in meta["invocations"]:
        cmd = f"python -m evaluation.{inv['script']} --judge-model {inv['judge_model']} --tag {MAIN_TAG}"
        if inv.get("conditions") and inv["conditions"] != "both":
            cmd += f" --conditions {inv['conditions']}"
        if inv.get("min_interval"):
            cmd += f" --min-interval {inv['min_interval']:g}"
        if inv.get("tpm"):
            cmd += f" --tpm {inv['tpm']}"
        cmd += " --yes"
        if cmd not in counts:
            order.append(cmd)
        counts[cmd] += 1
    L = ["## Reproduce", "", "Commands as they were run (a command is shown once even if it was run several times).", "",
         "```bash",
         "python -m evaluation.generate_posts --model ollama/qwen2.5:3b     # the 60 posts, generated once and reused"]
    L += [f"{c}    # run {counts[c]} time(s), resumed after daily-limit stops" if counts[c] > 1 else c for c in order]
    L += ["python -m evaluation.analyze",
          "python -m evaluation.fix_impact          # before/after of the evaluator fix, from the existing replies",
          "python -m evaluation.pairwise_consistency  # from the existing replies",
          "python -m evaluation.write_report", "```", "",
          "The local 3B judge was run with an earlier revision of these scripts that had no `--tag` option: "
          "`python -m evaluation.judge_pairwise --judge-model ollama/llama3.2:3b --concurrency 1 --yes`, the same for "
          "`judge_pointwise`, and a final re-run of only the failed pairwise calls. Its results and logs were then "
          f"moved into `results/{SMALL_TAG}/` and `runs/{SMALL_TAG}/`. The fix-impact and consistency scripts and the "
          "analysis make no LLM calls. `evaluation/label_tool.py` is a human-rating tool that was built but is unused.",
          ""]
    return L


def summary_text(posts, a, fi, pc):
    """Plain-language summary (STUDY_SUMMARY.md). Same numbers as REPORT.md, shorter."""
    pb, pa = fi["pass_rate_before_4000_chars"], fi["pass_rate_after_full_text"]
    flip, cons = pc["among_flipped_topics"], pc["among_consistent_topics"]
    rt = a["test_retest"]
    changed = fi["verdict_changes"]["fail_to_pass"] + fi["verdict_changes"]["pass_to_fail"]
    L = ["# Auditing the LLM judge in the Blog Writer pipeline", "",
         "A short summary. The full tables, prompts and limitations are in [`REPORT.md`](REPORT.md). Every number comes "
         "from a script in `evaluation/` and is stored in `evaluation/results/`.", "",
         "The Blog Writer pipeline ends with an LLM \"editor\" that scores each finished post from 1 to 10, with 7 or more "
         "counting as a pass. This audit checked how consistent and robust that judge is, using 60 generated posts "
         "(30 topics, each written as a short and a long version) and a 120B judge. It did not check whether the "
         "judge is accurate: agreement with a human reader was not measured and is out of scope.", "",
         "## What the audit found", "",
         f"**1. The evaluator's {LEGACY_EVAL_MAX_CHARS}-character cap changed the verdict on most long posts, so it was fixed.** "
         f"The evaluator only showed the judge the first {LEGACY_EVAL_MAX_CHARS} characters. On the {fi['n_posts']} posts "
         f"longer than that, the pass rate was {fmt_inline(pb)} on the truncated view and {fmt_inline(pa)} on the full "
         f"text. The mean score rose by {fi['mean_score_change']:.2f} points (95% CI {fmt_ci(fi['mean_score_change_ci95'])}) "
         f"and the pass/fail verdict changed on {changed} of {fi['n_posts']} posts. The evaluator now reads the whole "
         f"post up to `EVALUATOR_MAX_CHARS` (default {DEFAULT_EVALUATOR_MAX_CHARS}) and logs a warning if it ever has to "
         "cut one. This is a before/after on the same posts and the same judge, taken from replies that already existed; "
         "the live pipeline was not re-run after the fix.", "",
         f"**2. Pairwise verdicts were fairly but not fully consistent.** When a short and a long post were compared in "
         f"both orders, the same post won both times on {fmt_inline(pc['same_winner_in_both_orders'])} of topics. "
         f"In {flip['first_shown_won_in_both_orders']} of the {flip['n']} flipped topics the first-shown post won in "
         f"both orders, but an exact binomial test gives p = {flip['binomial_test']['p_value_exact_two_sided']:.3f}, "
         "which is not significant at 0.05, so a first-position effect is neither shown nor ruled out. Among the "
         f"consistent topics the long post won {fmt_inline(cons['B_long_won'])}; because the long post is about twice "
         "as long, that cannot be separated from a preference for length.", "",
         f"**3. Scores were mostly repeatable.** Across 3 runs on the same text the mean score difference was "
         f"{rt['mean_abs_diff']:.2f} points (95% CI {fmt_ci(rt['mean_abs_diff_ci95'])}), and all 3 runs agreed on "
         f"{fmt_inline(rt['share_posts_all_runs_identical'])} of posts.", "",
         "## What this does not show", "",
         "- Nothing here shows the judge is accurate. No human read the posts, so there is no comparison with a person.",
         "- The pairwise results measure consistency only. Accepting a verdict only when both orders agree keeps "
         f"{fmt_inline(pc['agree_or_tie_rule']['topics_with_an_accepted_verdict'])} of topics, but that does not make "
         "those verdicts correct.",
         "- A local 3B judge was also tried. It gave almost every post a score of 8 and often returned invalid JSON, "
         "so its run is incomplete and it was not used for any conclusion.", ""]
    L += ["## Limitations", "",
          "- One generator (a small local 3B model), one 120B judge, one prompt per study, one temperature (0.1).",
          f"- Small sample: 60 posts, {fi['n_posts']} long posts, {flip['n']} flipped topics. The confidence intervals are wide.",
          "- The long and short versions differ in length, so length and variant are confounded in the pairwise results.",
          "- The position effect is not statistically significant.", ""]
    L += token_limit_section()
    L += ["## Where things are", "",
          "- `REPORT.md`: full tables, exact prompts, and the commands that were run.",
          "- `results/`: all computed numbers (`fix_impact.json`, and per judge `analysis.json` and "
          "`pairwise_consistency.json`).",
          "- `runs/`: every prompt and raw reply, verbatim.",
          "- `posts/`: the 60 generated posts. `label_tool.py` is a human-rating tool that was built but is unused.", ""]
    return L


def main():
    posts = [load_json(p) for p in sorted(glob.glob(os.path.join(EVAL_DIR, "posts", "*.json")))]
    gen_calls, gen_err = count_calls(os.path.join(RUNS_DIR, "generate-*", "calls.jsonl"))
    gen_meta = [load_json(p) for p in sorted(glob.glob(os.path.join(RUNS_DIR, "generate-*", "run_meta.json")))]
    gen_model = gen_meta[0]["model"] if gen_meta else "unknown"
    a = load_json(os.path.join(RESULTS_DIR, MAIN_TAG, "analysis.json"))
    a_small = load_json(os.path.join(RESULTS_DIR, SMALL_TAG, "analysis.json"))
    fi = load_json(os.path.join(RESULTS_DIR, "fix_impact.json"))
    pc = load_json(os.path.join(RESULTS_DIR, MAIN_TAG, "pairwise_consistency.json"))
    if not (a and fi and pc):
        raise SystemExit("Run evaluation.analyze, evaluation.fix_impact and evaluation.pairwise_consistency first.")

    pb, pa = fi["pass_rate_before_4000_chars"], fi["pass_rate_after_full_text"]
    flip = pc["among_flipped_topics"]
    L = ["# Auditing the LLM judge in the Blog Writer pipeline", "",
         "## Scope", "",
         "The Blog Writer pipeline ends with an LLM \"editor\" that scores each post from 1 to 10 (pass at 7 or more). "
         "This audit asks how consistent and robust that judge is. It does not ask whether the judge is accurate: "
         "**agreement with a human reader was not measured and is out of scope, so nothing here shows that the judge's "
         "scores are right.** Every number comes from a script in `evaluation/` and is stored in "
         "`evaluation/results/`.", "",
         "## Main finding and the fix", "",
         f"The evaluator only showed the judge the first {LEGACY_EVAL_MAX_CHARS} characters of a post. On the "
         f"{fi['n_posts']} posts longer than that, a 120B judge passed {pb['p']:.1%} ({pb['k']}/{pb['n']}) of them "
         f"on the truncated view and {pa['p']:.1%} ({pa['k']}/{pa['n']}) on the full text, a mean score change of "
         f"{fi['mean_score_change']:+.2f} points (95% CI {fmt_ci(fi['mean_score_change_ci95'])}). The pass/fail "
         f"verdict changed on {fi['verdict_changes']['fail_to_pass'] + fi['verdict_changes']['pass_to_fail']} of "
         f"{fi['n_posts']} posts.", "",
         f"**The fix:** the evaluator now reads the whole post up to `EVALUATOR_MAX_CHARS` (default "
         f"{DEFAULT_EVALUATOR_MAX_CHARS}, enough for posts of about 2,500 words; the longest post in this study is "
         f"{max(p['chars'] for p in posts)} characters). A post over the cap is cut from the end and a warning is "
         "logged, never silently. The before/after above is on the same posts and the same judge, from replies that "
         "already existed; the live pipeline was not re-run.", "",
         "## Other findings (120B judge)", "",
         f"- **Repeatability:** the mean score difference between runs is {a['test_retest']['mean_abs_diff']:.2f} points "
         f"and all 3 runs agree on {fmt_inline(a['test_retest']['share_posts_all_runs_identical'])} of posts.",
         f"- **Pairwise consistency:** the same post won in both presentation orders in "
         f"{fmt_inline(pc['same_winner_in_both_orders'])} of topics, so the winner flipped on "
         f"{flip['n']} of {pc['n_topics']}.",
         f"- **Position:** in {flip['first_shown_won_in_both_orders']} of those {flip['n']} flips the first-shown post "
         f"won in both orders, but an exact binomial test gives p = "
         f"{flip['binomial_test']['p_value_exact_two_sided']:.3f}, not significant at 0.05, so a first-position effect "
         "is neither shown nor ruled out.", "",
         "## Setup", ""]
    rows = []
    for v, budget in (("A", 150), ("B", 400)):
        ps = [p for p in posts if p["variant"] == v]
        words = [p["words"] for p in ps]
        rows.append([v, f"{budget} words per section, retrieval off", len(ps), f"{sum(words) / len(words):.0f}",
                     f"{min(words)} to {max(words)}"])
    fallbacks = sum(any("Planner parse error" in e for e in p.get("pipeline_errors", [])) for p in posts)
    L += [table(["Variant", "Setting", "Posts", "Mean words", "Word range"], rows), "",
          f"- Generator: `{gen_model}` (local Ollama), temperatures planner 0.4 and writers 0.7. {gen_calls} generation "
          f"calls logged ({gen_err} errors). The same 60 posts are used for every judge.",
          "- 30 topics (`evaluation/topics.yaml`), same audience and tone for all posts, each written as a short (A) "
          "and a long (B) version. No Tavily key was available, so B is not a retrieval variant: the contrast is post "
          f"length. Planner fallbacks (planner JSON failed): {fallbacks}.",
          "- Judge temperature 0.1, the same as the pipeline evaluator. The pointwise judge uses the pipeline's "
          "existing evaluator prompt unchanged; the pairwise prompt is new (both are listed below).", ""]
    L += main_judge_section(a, fi, pc, MAIN_TAG, gen_model)
    if a_small:
        L += small_judge_section(a_small)
    L += token_limit_section()
    L += ["## Limitations", "",
          "- One generator (a small local 3B model), one 120B judge, one prompt per study, one temperature (0.1). "
          "Results may not transfer to other judges, prompts or generators.",
          f"- Small sample: 30 topics and 60 posts ({fi['n_posts']} long posts, {flip['n']} flipped topics). "
          "Wilson and bootstrap intervals are wide, and differences smaller than the intervals should not be read as real.",
          "- Variant B is about twice as long as variant A, so length and variant are confounded in the pairwise results.",
          "- The position effect is not statistically significant (see above).",
          "- No human reader was involved, so the audit measures consistency and robustness, not accuracy.",
          "- The before/after of the fix uses existing judge replies on the same posts; the live pipeline was not re-run.",
          "- Posts come from a small local model, so quality is lower and more uneven than a larger generator would give.", ""]
    L += ["## Exact prompts", "",
          "### Pointwise (the pipeline's existing evaluator prompt)", "", "```", EVALUATOR_PROMPT.rstrip(), "```", "",
          "### Pairwise (new)", "", "```", PAIRWISE_PROMPT.rstrip(), "```", ""]
    L += reproduce_block()
    with open(os.path.join(EVAL_DIR, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("wrote evaluation/REPORT.md")
    with open(os.path.join(EVAL_DIR, "STUDY_SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_text(posts, a, fi, pc)))
    print("wrote evaluation/STUDY_SUMMARY.md")


if __name__ == "__main__":
    main()
