import time
import re
from collections import defaultdict
from typing import List, Tuple, Any, Dict

import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# =====================================================
# Core MapReduce Logic
# =====================================================

def partition_records(records: List[Any], num_workers: int) -> List[List[Any]]:
    """Split records into num_workers contiguous chunks."""
    n = len(records)
    if num_workers <= 0:
        num_workers = 1
    chunk_size = max(1, (n + num_workers - 1) // num_workers)
    chunks: List[List[Any]] = []
    for i in range(0, n, chunk_size):
        chunks.append(records[i:i + chunk_size])
    return chunks


def wordcount_mapper(line: str) -> List[Tuple[str, int]]:
    """Map: line -> list of (word, 1)."""
    tokens = re.findall(r"\w+", line.lower())
    return [(t, 1) for t in tokens]


def wordcount_reducer(word: str, counts: List[int]) -> Tuple[str, int]:
    """Reduce: (word, [1,1,1,...]) -> (word, total_count)."""
    return word, sum(counts)


def simulated_worker_map_reduce(
    records: List[str],
    num_workers: int = 4,
    verbose: bool = False,
) -> Tuple[List[Tuple[str, int]], Dict[str, float], str, Dict[str, List[int]], List[Dict[str, int]]]:
    """
    Single-process simulation of MapReduce with 'num_workers' workers.

    Returns:
        mr_results: list of (word, count) sorted by count desc
        timings: dict of phase timings
        logs: string with worker logs
        grouped: dict key -> list of values (for visualization)
        worker_key_counts: list[dict] per worker, key -> count emitted
    """

    log_lines: List[str] = []
    timings: Dict[str, float] = {}

    # ---------------- Partition ----------------
    t0 = time.perf_counter()
    chunks = partition_records(records, num_workers)
    t1 = time.perf_counter()
    timings["partition"] = t1 - t0
    log_lines.append(f"Partitioned {len(records)} records into {len(chunks)} workers.")

    # ---------------- Map Phase ----------------
    t_map_start = time.perf_counter()
    all_intermediate: List[Tuple[str, int]] = []
    worker_key_counts: List[Dict[str, int]] = []

    for wid, chunk in enumerate(chunks):
        if verbose:
            log_lines.append(f"[Worker {wid}] received {len(chunk)} records")

        local_intermediate: List[Tuple[str, int]] = []
        key_count: Dict[str, int] = defaultdict(int)

        for rec in chunk:
            kvs = wordcount_mapper(rec)
            local_intermediate.extend(kvs)
            for k, v in kvs:
                key_count[k] += v

        worker_key_counts.append(dict(key_count))

        if verbose:
            log_lines.append(
                f"[Worker {wid}] emitted {len(local_intermediate)} kv pairs "
                f"({len(key_count)} distinct keys)"
            )

        all_intermediate.extend(local_intermediate)

    t_map_end = time.perf_counter()
    timings["map"] = t_map_end - t_map_start

    # ---------------- Shuffle Phase ----------------
    t_shuffle_start = time.perf_counter()
    grouped: Dict[str, List[int]] = defaultdict(list)
    for k, v in all_intermediate:
        grouped[k].append(v)
    t_shuffle_end = time.perf_counter()
    timings["shuffle"] = t_shuffle_end - t_shuffle_start
    log_lines.append(f"After shuffle: {len(grouped)} distinct keys")

    # ---------------- Reduce Phase ----------------
    t_reduce_start = time.perf_counter()
    results: List[Tuple[str, int]] = []
    for k, vals in grouped.items():
        results.append(wordcount_reducer(k, vals))
    t_reduce_end = time.perf_counter()
    timings["reduce"] = t_reduce_end - t_reduce_start

    timings["total"] = (
        timings["partition"] + timings["map"] + timings["shuffle"] + timings["reduce"]
    )

    # Sort results by count desc
    mr_results = sorted(results, key=lambda x: x[1], reverse=True)

    return mr_results, timings, "\n".join(log_lines), dict(grouped), worker_key_counts


def baseline_wordcount(records: List[str]) -> Tuple[Dict[str, int], float]:
    """Simple non-MapReduce baseline: one big dict."""
    t0 = time.perf_counter()
    counts: Dict[str, int] = defaultdict(int)
    for line in records:
        for token in re.findall(r"\w+", line.lower()):
            counts[token] += 1
    t1 = time.perf_counter()
    return counts, (t1 - t0)


# =====================================================
# Example & Synthetic Data
# =====================================================

EXAMPLES = {
    "Small Example 1": "The quick brown fox jumps over the lazy dog.\n" * 20,
    "Small Example 2": "MapReduce is a distributed system model.\n" * 50,
    "Medium Example": "Distributed systems scale by partitioning data.\n" * 200,
}


def make_synthetic_text(num_lines: int, vocab_size: int = 100) -> str:
    """Generate synthetic text for stress testing."""
    base_words = [f"word{i}" for i in range(vocab_size)]
    lines = []
    for i in range(num_lines):
        w1 = base_words[i % vocab_size]
        w2 = base_words[(i * 7) % vocab_size]
        w3 = base_words[(i * 13) % vocab_size]
        lines.append(f"{w1} {w2} {w3} map reduce distributed systems line{i}")
    return "\n".join(lines)


# =====================================================
# Streamlit UI
# =====================================================

st.set_page_config(page_title="MapReduce Playground", layout="wide")

st.title("MapReduce Simulation")
st.caption(
    "An interactive MapReduce sandbox where you can literally “see” the Map → Shuffle → Reduce pipeline happen. "
    "Upload your own text, generate synthetic datasets, or use built-in examples. "
    "Tune the number of simulated workers and watch how data flows, reorganizes, and aggregates across the system."
)

# ---------------- Sidebar: Input selection ----------------
st.sidebar.header("Input Data")
st.sidebar.caption("Adjust workers, run experiments, and **see** Map → Shuffle → Reduce in action.")

mode = st.sidebar.radio(
    "Choose input source",
    ["Example text", "Upload file", "Synthetic generator"],
)

raw_text = ""

if mode == "Example text":
    example_choice = st.sidebar.selectbox("Example dataset", list(EXAMPLES.keys()))
    raw_text = EXAMPLES[example_choice]

elif mode == "Upload file":
    uploaded_file = st.sidebar.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded_file is not None:
        raw_text = uploaded_file.read().decode("utf-8", errors="ignore")
    else:
        raw_text = ""

else:  # Synthetic generator
    num_lines = st.sidebar.slider("Number of lines", 100, 50000, 2000, step=100)
    vocab_size = st.sidebar.slider("Vocabulary size", 20, 2000, 200)
    if st.sidebar.button("Generate synthetic text"):
        raw_text = make_synthetic_text(num_lines=num_lines, vocab_size=vocab_size)
        st.session_state["synthetic_raw"] = raw_text
    else:
        raw_text = st.session_state.get("synthetic_raw", "")

if not raw_text.strip():
    st.warning("No text loaded yet. Choose an example, upload a file, or generate synthetic text.")
    st.stop()

records = [line.strip() for line in raw_text.split("\n") if line.strip()]
st.write(f"Loaded **{len(records)}** non-empty lines.")

# ---------------- Sidebar: Experiment settings ----------------
st.sidebar.header("Experiment Settings")
num_workers = st.sidebar.slider("Number of simulated workers (single run)", 1, 64, 4)
top_n = st.sidebar.slider("Top-N words to show", 5, 50, 20)
verbose_logs = st.sidebar.checkbox("Verbose worker logs", value=False)

st.sidebar.caption(
    "Created by [Alfyn](https://jaronchai.com). "
    "Contribute to the project on [Github](https://github.com/jarondlk/map-reduce-sim)! "
    "Reference: [MapReduce (Dean & Ghemawat, 2004)](https://storage.googleapis.com/gweb-research2023-media/pubtools/4449.pdf)."
)

# ---------------- Tabs ----------------
tab_single, tab_scaling = st.tabs(["Single Run (with visuals)", "Scaling Experiment"])


# =====================================================
# Tab 1: Single Run with Visualizations
# =====================================================
with tab_single:
    st.subheader("Single Run: Baseline vs MapReduce")

    # Run / Reset buttons (stacked)
    run_single = st.button("Run single experiment")
    reset_single = st.button("Reset experiment")

    if reset_single:
        if "single_results" in st.session_state:
            del st.session_state["single_results"]
        st.rerun()

    # Pipeline diagram at top
    st.markdown("### MapReduce Pipeline")
    pipeline_diagram = """
    digraph {
        rankdir=LR;

        Records [shape=box, style=filled, fillcolor=lightgray, label="Input Records"];
        Map [shape=box, style=filled, fillcolor=lightblue, label="Map (per worker)"];
        Shuffle [shape=box, style=filled, fillcolor=lightgreen, label="Shuffle / Group by Key"];
        Reduce [shape=box, style=filled, fillcolor=orange, label="Reduce (per key)"];
        Output [shape=box, style=filled, fillcolor=gold, label="Final Word Counts"];

        Records -> Map;
        Map -> Shuffle;
        Shuffle -> Reduce;
        Reduce -> Output;
    }
    """
    st.graphviz_chart(pipeline_diagram)

    st.markdown("---")

    if run_single:
        with st.spinner("Running baseline and MapReduce..."):
            baseline_counts, baseline_time = baseline_wordcount(records)
            mr_results, mr_timings, logs, grouped, worker_key_counts = simulated_worker_map_reduce(
                records=records,
                num_workers=num_workers,
                verbose=verbose_logs,
            )

        st.session_state["single_results"] = {
            "baseline_counts": baseline_counts,
            "baseline_time": baseline_time,
            "mr_results": mr_results,
            "mr_timings": mr_timings,
            "logs": logs,
            "grouped": grouped,
            "worker_key_counts": worker_key_counts,
            "num_workers": num_workers,
            "num_records": len(records),
        }

    if "single_results" not in st.session_state:
        st.info("Click **Run single experiment** to compute timings and see visualizations.")
    else:
        res = st.session_state["single_results"]
        baseline_counts = res["baseline_counts"]
        baseline_time = res["baseline_time"]
        mr_results = res["mr_results"]
        mr_timings = res["mr_timings"]
        logs = res["logs"]
        grouped = res["grouped"]
        worker_key_counts = res["worker_key_counts"]
        used_workers = res["num_workers"]

        st.markdown("### Timings & Overview")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Baseline (no MapReduce)", f"{baseline_time:.6f} s")
            st.write(f"Distinct words (baseline): **{len(baseline_counts)}**")

        with col2:
            st.metric("MapReduce total time", f"{mr_timings['total']:.6f} s")
            st.json(mr_timings)

        with col3:
            if baseline_time > 0:
                ratio = mr_timings["total"] / baseline_time
                st.metric("MR / Baseline time ratio", f"{ratio:.3f}×")
            st.write(f"Workers used: **{used_workers}**")
            st.write(f"Records: **{res['num_records']}**")

        st.markdown("---")
        st.markdown("## Map Phase")

        # Map: KV pairs per worker
        worker_kv_rows = []
        for wid, wc in enumerate(worker_key_counts):
            total_kv = sum(wc.values())
            worker_kv_rows.append(
                {"worker": f"Worker {wid}", "kv_pairs": total_kv, "distinct_keys": len(wc)}
            )

        df_map_vis = pd.DataFrame(worker_kv_rows)
        c1, c2 = st.columns(2)
        with c1:
            st.bar_chart(df_map_vis.set_index("worker")[["kv_pairs"]])
            st.caption("Each worker emits this many key-value pairs during the map phase.")
        with c2:
            st.table(df_map_vis)

        # ---------------- Shuffle Visualization ----------------
        st.markdown("---")
        st.markdown("## Shuffle Phase (Workers → Keys)")

        # Build links for Sankey: worker -> key, but limit to top keys
        max_keys_for_sankey = 10
        top_keys_for_sankey = {w for w, c in mr_results[:max_keys_for_sankey]}

        sankey_links = []
        all_labels = []

        # Workers as sources
        worker_labels = [f"Worker {i}" for i in range(len(worker_key_counts))]
        all_labels.extend(worker_labels)

        # Keys as targets
        key_labels = list(top_keys_for_sankey)
        all_labels.extend(key_labels)

        label_to_idx = {lab: idx for idx, lab in enumerate(all_labels)}

        for wid, wc in enumerate(worker_key_counts):
            worker_name = f"Worker {wid}"
            for key, count in wc.items():
                if key not in top_keys_for_sankey:
                    continue
                sankey_links.append(
                    dict(
                        source=label_to_idx[worker_name],
                        target=label_to_idx[key],
                        value=count,
                    )
                )

        if sankey_links:
            fig = go.Figure(data=[go.Sankey(
                arrangement="snap",
                node=dict(
                    label=all_labels,
                    pad=15,
                    thickness=18,
                ),
                link=dict(
                    source=[l["source"] for l in sankey_links],
                    target=[l["target"] for l in sankey_links],
                    value=[l["value"] for l in sankey_links],
                ),
            )])
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Keys flow from workers (left) into shuffled key groups (right).")
        else:
            st.info("Not enough data to build a shuffle visualization (Sankey).")

        # ---------------- Reduce Visualization ----------------
        st.markdown("---")
        st.markdown("## Reduce Phase (Final Word Counts)")

        df_top = pd.DataFrame(mr_results[:top_n], columns=["word", "count"])

        c3, c4 = st.columns(2)
        with c3:
            st.subheader(f"Top {top_n} words (table)")
            st.table(df_top)

        with c4:
            st.subheader(f"Top {top_n} words (bar chart)")
            st.bar_chart(df_top.set_index("word"))

        # ---------------- Logs ----------------
        if verbose_logs:
            st.markdown("---")
            with st.expander("Worker logs"):
                st.text(logs)


# =====================================================
# Tab 2: Scaling Experiment
# =====================================================
with tab_scaling:
    st.subheader("Scaling Experiment: MapReduce vs Number of Workers")

    max_workers = st.slider("Max workers to test", 1, 64, 16)

    run_scaling = st.button("Run scaling experiment")

    if run_scaling:
        with st.spinner("Running scaling experiment..."):
            baseline_counts, baseline_time = baseline_wordcount(records)

            worker_values = list(range(1, max_workers + 1))
            rows = []

            progress = st.progress(0.0)
            for i, w in enumerate(worker_values):
                mr_results, mr_timings, _, _, _ = simulated_worker_map_reduce(
                    records=records,
                    num_workers=w,
                    verbose=False,
                )
                rows.append({
                    "workers": w,
                    "mapreduce_total": mr_timings["total"],
                    "map_time": mr_timings["map"],
                    "shuffle_time": mr_timings["shuffle"],
                    "reduce_time": mr_timings["reduce"],
                })
                progress.progress((i + 1) / len(worker_values))

        df = pd.DataFrame(rows).sort_values("workers")
        base_mr_1 = df.loc[df["workers"] == 1, "mapreduce_total"].iloc[0]
        df["speedup_vs_mr1"] = base_mr_1 / df["mapreduce_total"]
        df["baseline_time"] = baseline_time
        df["speedup_vs_baseline"] = baseline_time / df["mapreduce_total"]

        st.markdown("### Total MapReduce Time vs Workers")
        st.line_chart(df.set_index("workers")[["mapreduce_total"]])

        st.markdown("### Speedup vs 1-Worker MapReduce")
        st.line_chart(df.set_index("workers")[["speedup_vs_mr1"]])

        st.markdown("### Speedup vs Baseline (no MapReduce)")
        st.line_chart(df.set_index("workers")[["speedup_vs_baseline"]])

        st.markdown("### Raw Data")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Set the max workers and click **Run scaling experiment** to see scaling behavior.")
