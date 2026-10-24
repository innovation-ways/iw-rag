# ruff: noqa: E501
"""Plain-language explanations of every number and setting on the benchmark pages.

One source of truth, used twice: the benchmark pages show it in their hover tooltips and
leaderboard headers, and the Learn section's "Reading the benchmarks" lesson shows it as a
decoder.  Each entry says what the number means in everyday words, gives a concrete
example of how to read it, and says which direction is better.  The keys match the metric
keys the benchmark pages already use (``recall_at_5``, ``p99`` latencies, the six
leaderboards, the A/B statistics), so a metric added there without an explanation here is
caught by a test.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Term:
    """One benchmark number or setting, explained for someone new to it.

    Attributes:
        key: The key the benchmark pages use for it.
        name: A friendly name.
        group: Which part of the pages it belongs to.
        plain: What it means, in everyday words.
        example: A concrete reading of a value.
        good: Which direction is better, and roughly what a good value looks like.
        watch: A caveat worth knowing, or ``""``.
    """

    key: str
    name: str
    group: str
    plain: str
    example: str
    good: str
    watch: str = ""


GROUPS: tuple[str, ...] = (
    "Did it find the right passages?",
    "Is the answer honest?",
    "How fast is it?",
    "How much GPU memory?",
    "Comparing two runs",
    "The leaderboards",
    "Run settings",
)

_RET, _ANS, _SPEED, _MEM, _CMP, _BOARD, _SET = GROUPS

TERMS: tuple[Term, ...] = (
    # ----------------------------------------------------------------- retrieval
    Term(
        "recall_at_5",
        "Recall at 5",
        _RET,
        "Did a correct passage show up in the first five results?",
        "0.80 means that for 8 questions out of 10, a correct passage was among the top five.",
        "Higher is better. 1.0 means it never missed.",
        "It ignores the order inside the top five and says nothing about the final answer.",
    ),
    Term(
        "recall_at_10",
        "Recall at 10",
        _RET,
        "The same question as recall at 5, but allowing the first ten results.",
        "0.90 means a correct passage was in the top ten for 9 questions out of 10.",
        "Higher is better. It is never lower than recall at 5.",
    ),
    Term(
        "ndcg_at_5",
        "Search quality (nDCG at 5)",
        _RET,
        "Like recall, but it also rewards putting the correct passage near the top.",
        "A correct passage in 1st place scores 1.0, in 3rd place about 0.5, and outside the top five 0.",
        "Higher is better. 1.0 is a perfect ordering.",
        "The value is averaged over all questions, so 0.6 does not mean 'always third'.",
    ),
    Term(
        "ndcg_at_10",
        "Search quality (nDCG at 10)",
        _RET,
        "The same ordering score, looking at the first ten results.",
        "A correct passage in 1st place scores 1.0, in 3rd place about 0.5, in 10th place about 0.29.",
        "Higher is better.",
    ),
    Term(
        "mrr",
        "Mean reciprocal rank (MRR)",
        _RET,
        "How high the first correct passage sits, on average.",
        "A correct first result counts 1, a second place counts 0.5, a third 0.33. An MRR of 0.85 means the first correct result is usually first or second.",
        "Higher is better.",
    ),
    # ------------------------------------------------------------------- answers
    Term(
        "faithfulness",
        "Faithfulness",
        _ANS,
        "Does the answer stick to its sources? It is the share of claims in the answer that the retrieved passages really support.",
        "0.90 means about 9 in 10 claims were backed by the documents, and about 1 in 10 was invented.",
        "Higher is better. It is the closest thing to a 'does it make things up?' score.",
        "A refusal such as 'I do not have enough information' is trivially faithful, so a high score can also mean the system refused a lot. A judge model scores it.",
    ),
    Term(
        "context_precision",
        "Context precision",
        _ANS,
        "How much of what the model was given was actually useful.",
        "0.50 means half of the passages handed to the model were noise.",
        "Higher is better.",
    ),
    Term(
        "faithfulness_kappa",
        "Judge agreement (kappa)",
        _ANS,
        "Does the judge model agree with human labels about which answers are faithful?",
        "A kappa near 1 means they agree almost always, and near 0 means no better than chance.",
        "Higher is better. It tells you how far to trust the faithfulness score.",
        "Only filled in when a separate harness has run; blank is normal.",
    ),
    Term(
        "context_recall",
        "Context recall",
        _ANS,
        "Did the retrieved passages contain the facts the correct answer needs?",
        "0.75 means three quarters of the facts in the reference answer were present in what was retrieved.",
        "Higher is better.",
        "Needs a dataset that ships reference answers.",
    ),
    Term(
        "factual_correctness",
        "Factual correctness",
        _ANS,
        "How closely the answer matches a known-correct reference answer.",
        "0.80 means the answer agrees with the reference on about 80% of its facts.",
        "Higher is better.",
        "Needs reference answers.",
    ),
    Term(
        "noise_sensitivity",
        "Noise sensitivity",
        _ANS,
        "How often irrelevant passages misled the answer.",
        "0.20 means about one claim in five came from a passage that should have been ignored.",
        "LOWER is better.",
        "Needs reference answers.",
    ),
    # --------------------------------------------------------------------- speed
    Term(
        "model_load_ms",
        "Model load time",
        _SPEED,
        "How long it took to load the model onto the GPU before the run.",
        "10 s means the card waited ten seconds for the model; no question was being answered in that time.",
        "Lower is better, but it is kept apart from answer time on purpose.",
        "Never added to the per-question or end-to-end times.",
    ),
    Term(
        "retrieve_generate_ms",
        "Time per question",
        _SPEED,
        "On average, how long one question took to search and answer.",
        "2.4 s means the average question took about two and a half seconds.",
        "Lower is better.",
    ),
    Term(
        "ttft_p99_s",
        "Wait for the first word (p99)",
        _SPEED,
        "How long until the first word of the answer appears, in the slow cases. 'p99' means 99 requests in 100 were faster than this.",
        "0.8 s means 99 in 100 answers started within 0.8 seconds.",
        "Lower is better.",
        "p99 shows the slow cases, not the average. It is blank when the server's metrics were not scraped.",
    ),
    Term(
        "tpot_p99_s",
        "Time between words (p99)",
        _SPEED,
        "Once the answer starts streaming, how long each new word piece takes, in the slow cases.",
        "0.02 s per token is about 50 tokens a second.",
        "Lower is better.",
    ),
    Term(
        "e2e_p99_s",
        "Worst-case wait (p99)",
        _SPEED,
        "The full time from question to finished answer, in the slow cases: 99 in 100 requests finished faster.",
        "5 s means 99 in 100 answers were complete within five seconds.",
        "Lower is better.",
    ),
    Term(
        "queue_time_p99_s",
        "Time spent waiting in line (p99)",
        _SPEED,
        "How long a request waited before the server started on it, in the slow cases.",
        "0.1 s means almost nothing queued up.",
        "Lower is better.",
    ),
    # -------------------------------------------------------------------- memory
    Term(
        "vram_peak_mb",
        "GPU memory peak",
        _MEM,
        "The most video memory in use at any moment during the run.",
        "28 GB out of the card's 32 GB leaves little room for anything else.",
        "Lower leaves more room. It must stay under the card's 32 GB.",
    ),
    Term(
        "vram_baseline_mb",
        "GPU memory before the run",
        _MEM,
        "What was already in use on the card before this setup was loaded.",
        "2 GB means other programs were already using two gigabytes.",
        "Only for context.",
    ),
    Term(
        "vram_incremental_mb",
        "GPU memory this setup needs",
        _MEM,
        "The peak minus what was already used: the memory this particular setup took.",
        "20 GB means the chosen models needed about twenty gigabytes by themselves.",
        "Lower is better.",
    ),
    Term(
        "util_peak_pct",
        "GPU busy (peak)",
        _MEM,
        "The highest share of the GPU's compute power in use at any moment.",
        "95% means the card was nearly saturated at its busiest.",
        "Neither high nor low is 'good': it shows whether the card was the bottleneck.",
    ),
    Term(
        "util_mean_pct",
        "GPU busy (average)",
        _MEM,
        "The average share of the GPU's compute power in use during the run.",
        "40% means the card was idle more than half of the time, so something else was the limit.",
        "Neither high nor low is 'good'.",
    ),
    Term(
        "prefix_cache_hit_rate",
        "Reused work (prefix cache)",
        _MEM,
        "How often the server reused work it had already done for a prompt that starts the same way as an earlier one.",
        "70% means seven in ten prompt tokens did not have to be recomputed.",
        "Higher is cheaper.",
    ),
    Term(
        "kv_cache_peak_pct",
        "Working memory used (KV cache)",
        _MEM,
        "How full the server's working memory for the conversation got.",
        "95% means the server nearly ran out and was limited by memory, not by compute.",
        "Lower leaves more headroom for more users.",
    ),
    Term(
        "generation_tokens_delta",
        "Words written (tokens)",
        _MEM,
        "The total number of word pieces (tokens) the model wrote during the run. A token is roughly three quarters of a word.",
        "10,000 tokens is about 7,500 words.",
        "Only for context: it is used to work out writing speed.",
    ),
    # --------------------------------------------------------------- comparing
    Term(
        "delta",
        "Difference (B minus A)",
        _CMP,
        "How much better or worse run B is than run A on this metric.",
        "On a 0 to 1 score, +0.05 means B scored five points higher. For 'lower is better' metrics a negative number is the improvement.",
        "Read it together with the confidence interval.",
    ),
    Term(
        "p_value",
        "p-value",
        _CMP,
        "How surprising a gap this big would be if the two setups were really equally good. Small means the gap is unlikely to be luck.",
        "0.01 means a gap this large would turn up by chance about once in a hundred times.",
        "Smaller is stronger evidence.",
        "It says nothing about how big or useful the gap is.",
    ),
    Term(
        "p_corrected",
        "Corrected p-value",
        _CMP,
        "The p-value adjusted because many metrics are tested at once; test enough metrics and one will look significant by luck.",
        "A raw p-value of 0.03 can become 0.12 once corrected: no longer convincing.",
        "Smaller is stronger evidence. This is the one to trust.",
        "It uses the Bonferroni-Holm method.",
    ),
    Term(
        "ci_95",
        "Confidence range (95%)",
        _CMP,
        "The range in which the true difference very probably lies, given the questions asked.",
        "A difference of +0.04 with a range of -0.02 to +0.10 could be zero, so it may be noise.",
        "If the range includes 0, do not trust the difference.",
    ),
    Term(
        "significant",
        "Significant",
        _CMP,
        "A difference counts as significant when it passes the corrected test and its confidence range stays on one side of zero.",
        "'not significant' does not mean 'no difference': it means this data cannot tell.",
        "Look for ✓ significant before believing a gap.",
    ),
    Term(
        "underpowered",
        "Underpowered",
        _CMP,
        "Too few questions to reliably tell a small difference from chance.",
        "With about 37 questions, only fairly large gaps can be detected.",
        "Treat an underpowered result as a hint, not a verdict.",
    ),
    Term(
        "winner",
        "Winner",
        _CMP,
        "A setup is crowned only when it beats the other on both search quality (nDCG at 5) and honesty (faithfulness).",
        "A split result, better at search but worse on honesty, reports no winner on purpose.",
        "A missing winner is a legitimate result.",
    ),
    Term(
        "baseline",
        "Baseline",
        _CMP,
        "The run every other run is compared against.",
        "With baseline A, 'B: +0.05' means B beat A by five points.",
        "Pick the setup you already use as the baseline.",
    ),
    # ------------------------------------------------------------- leaderboards
    Term(
        "tok_per_s",
        "Writing speed (tokens per second)",
        _BOARD,
        "How many word pieces (tokens) the model writes each second. A token is roughly three quarters of a word.",
        "50 tokens a second is about 37 words a second, faster than anyone reads.",
        "Higher is faster.",
    ),
    Term(
        "latency",
        "Worst-case wait",
        _BOARD,
        "The leaderboard for the worst-case answer time (end-to-end p99, in seconds).",
        "5 s means 99 in 100 answers were complete within five seconds.",
        "LOWER is better, so the fastest run is listed first.",
    ),
    Term(
        "vram_efficiency",
        "Quality per gigabyte",
        _BOARD,
        "How much search quality you get for each gigabyte of GPU memory: nDCG at 5 divided by peak memory in GiB.",
        "0.04 means each gigabyte of memory bought 0.04 points of search quality.",
        "Higher is better.",
        "It favours small setups that are still good, not simply the best one.",
    ),
    Term(
        "composite",
        "All-round score",
        _BOARD,
        "One number that blends the five other leaderboards: honesty 30%, search quality 20%, writing speed 20%, worst-case wait 15% and quality per gigabyte 15%. Each part is scaled to 0 to 1 across the runs being ranked.",
        "0.80 means the run sits near the top of most boards.",
        "Higher is better.",
        "The weights are a judgement call; a different use may care more about speed than honesty.",
    ),
    # ------------------------------------------------------------------ settings
    Term(
        "generator",
        "Generator",
        _SET,
        "The model that writes the answer.",
        "A bigger generator is usually more fluent but slower and needs more GPU memory.",
        "Pick what you would actually deploy.",
    ),
    Term(
        "embedder",
        "Embedder",
        _SET,
        "The model that turns questions and passages into numbers so that similar meanings can be found.",
        "Changing it means re-embedding every document.",
        "Keep it the same unless you are testing it.",
    ),
    Term(
        "reranker",
        "Reranker",
        _SET,
        "The model that re-reads the best search results and puts the most relevant first.",
        "It mostly improves ordering, which shows up in nDCG and MRR.",
        "Switch it to see what it buys.",
    ),
    Term(
        "top_k",
        "Passages kept (top_k)",
        _SET,
        "How many passages are passed to the generator after reranking.",
        "5 means the model reads five passages. More gives it more to work with, but adds noise and cost.",
        "Between 5 and 10 is typical.",
    ),
    Term(
        "mode",
        "Retrieval mode",
        _SET,
        "Which retrieval route to use. 'auto' lets the router choose for each question; the others force one route.",
        "'graph' forces the knowledge-graph search for every question.",
        "Leave on auto unless you are testing one route.",
    ),
    Term(
        "dataset",
        "Dataset (golden set)",
        _SET,
        "A fixed list of questions with known correct passages, used to measure every setup the same way.",
        "the project's benchmark-golden set, for example, has about 37 questions.",
        "Only compare runs that used the same dataset.",
    ),
    Term(
        "judge",
        "Judge",
        _SET,
        "The model that grades answers, for example whether each claim is supported. It is a different family from the generator, so it does not favour its own style.",
        "A judge that is down leaves the answer-quality scores blank.",
        "Keep it fixed between runs you want to compare.",
    ),
    Term(
        "vram_budget",
        "GPU memory budget",
        _SET,
        "The three chosen models must fit in the card's 32 GB together. The bar shows how much they would use.",
        "A run that does not fit is refused before it starts.",
        "Leave some headroom.",
    ),
)

_BY_KEY = {t.key: t for t in TERMS}


def get_term(key: str) -> Term | None:
    """Return the term for ``key``, or ``None`` if there is none."""
    return _BY_KEY.get(key)


def plain_map() -> dict[str, dict[str, str]]:
    """The explanations as plain dicts, ready to hand to the benchmark pages' script."""
    return {
        t.key: {
            "name": t.name,
            "plain": t.plain,
            "example": t.example,
            "good": t.good,
            "watch": t.watch,
        }
        for t in TERMS
    }


__all__ = ["GROUPS", "TERMS", "Term", "get_term", "plain_map"]
