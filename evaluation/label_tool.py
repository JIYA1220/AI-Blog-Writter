"""
Human labeling tool (Streamlit). Run:  streamlit run evaluation/label_tool.py

- Shows single posts (rate 1-10) and A/B pairs (which is better?) in a fixed random order.
- In pairs, which variant appears as "Post 1" is randomized per topic.
- Shows no judge scores, variant names, word budgets or metadata.
- Every answer is saved at once to evaluation/human_labels.csv; closing the browser and coming back resumes
  at the next unanswered item. "Undo last answer" removes your most recent row.

The plan (which 30 posts, the item order, the pair orientation) is created once with a fixed seed and saved
to evaluation/label_plan.json. It contains no judge output.
"""

import csv
import datetime
import json
import os
import random
import sys

import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evaluation.judge_common import EVAL_DIR, load_posts  # noqa: E402

PLAN_PATH = os.path.join(EVAL_DIR, "label_plan.json")
LABELS_PATH = os.path.join(EVAL_DIR, "human_labels.csv")
FIELDS = ["item_type", "item_id", "rating", "choice_slot", "choice_variant", "first_variant", "timestamp"]
SEED = 20260101
N_SINGLES = 30


def build_plan(posts: dict) -> dict:
    rng = random.Random(SEED)
    by_variant = {v: sorted(p for p in posts if posts[p]["variant"] == v) for v in ("A", "B")}
    half = N_SINGLES // 2
    singles = rng.sample(by_variant["A"], min(half, len(by_variant["A"]))) + \
        rng.sample(by_variant["B"], min(half, len(by_variant["B"])))
    topics = sorted({p["topic_id"] for p in posts.values()
                     if f"{p['topic_id']}_A" in posts and f"{p['topic_id']}_B" in posts})
    items = [{"type": "single", "id": pid} for pid in singles]
    items += [{"type": "pair", "id": t, "first_variant": rng.choice(["A", "B"])} for t in topics]
    rng.shuffle(items)
    return {"seed": SEED, "items": items}


def load_plan(posts: dict) -> dict:
    if os.path.exists(PLAN_PATH):
        with open(PLAN_PATH, encoding="utf-8") as f:
            return json.load(f)
    plan = build_plan(posts)
    with open(PLAN_PATH, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)
    return plan


def read_labels() -> list:
    if not os.path.exists(LABELS_PATH):
        return []
    with open(LABELS_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_labels(rows: list):
    with open(LABELS_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def append_label(row: dict):
    new = not os.path.exists(LABELS_PATH)
    with open(LABELS_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


st.set_page_config(page_title="Blog labeling", layout="wide")
posts = load_posts()
if not posts:
    st.error("No posts found in evaluation/posts/. Run evaluation/generate_posts.py first.")
    st.stop()

plan = load_plan(posts)
labels = read_labels()
answered = {(r["item_type"], r["item_id"]) for r in labels}
remaining = [it for it in plan["items"] if (it["type"], it["id"]) not in answered]

st.sidebar.markdown("### Progress")
st.sidebar.progress(len(answered) / len(plan["items"]))
st.sidebar.write(f"{len(answered)} of {len(plan['items'])} done")
if labels and st.sidebar.button("Undo last answer"):
    write_labels(labels[:-1])
    st.rerun()

if not remaining:
    st.success("All items labeled. Thank you! You can close this tab.")
    st.stop()

item = remaining[0]
st.caption(f"Item {len(answered) + 1} of {len(plan['items'])}")

if item["type"] == "single":
    post = posts[item["id"]]
    st.subheader("Rate this blog post")
    st.write(f"**Topic:** {post['topic']}  \n**Audience:** {post['audience']}  \n**Tone:** {post['tone']}")
    st.markdown("---")
    st.markdown(post["content"])
    st.markdown("---")
    rating = st.slider("Overall quality, 1 (poor) to 10 (excellent)", 1, 10, 5, key=f"r_{item['id']}")
    st.caption("Judge by relevance, tone, flow and depth. A score of 7 or more will count as 'pass'.")
    if st.button("Save and next", type="primary"):
        append_label({"item_type": "single", "item_id": item["id"], "rating": rating, "choice_slot": "",
                      "choice_variant": "", "first_variant": "",
                      "timestamp": datetime.datetime.now().isoformat(timespec="seconds")})
        st.rerun()
else:
    first_v = item["first_variant"]
    second_v = "B" if first_v == "A" else "A"
    first, second = posts[f"{item['id']}_{first_v}"], posts[f"{item['id']}_{second_v}"]
    st.subheader("Which post is better?")
    st.write(f"**Topic:** {first['topic']}  \n**Audience:** {first['audience']}  \n**Tone:** {first['tone']}")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("## Post 1")
        st.markdown(first["content"])
    with c2:
        st.markdown("## Post 2")
        st.markdown(second["content"])
    st.markdown("---")
    choice = st.radio("Overall, which is better?", ["Post 1", "Post 2", "Tie"], index=None,
                      key=f"p_{item['id']}", horizontal=True)
    if st.button("Save and next", type="primary", disabled=choice is None):
        slot = {"Post 1": "first", "Post 2": "second", "Tie": "tie"}[choice]
        variant = {"first": first_v, "second": second_v, "tie": "tie"}[slot]
        append_label({"item_type": "pair", "item_id": item["id"], "rating": "", "choice_slot": slot,
                      "choice_variant": variant, "first_variant": first_v,
                      "timestamp": datetime.datetime.now().isoformat(timespec="seconds")})
        st.rerun()
