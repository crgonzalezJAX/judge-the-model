"""Starter app for Streamlit Community Cloud: three plain sections that read results/ and show correct numbers.

Run locally:  bash run_app.sh  (your agent writes it; it runs streamlit run streamlit_app.py)
Build each section up to its required minimum (see the activity website). Keep every number
coming from m11.py, and label every table and plot "Training" or "Test".
"""

import pandas as pd
import streamlit as st

import m11

TRAIN = "Training"
TEST = "Test"
SECTIONS = ["1. Beyond Accuracy", "2. A Threshold for the Job", "3. Training vs. Test"]

st.set_page_config(page_title="Judge the Model", layout="wide")


@st.cache_data
def load(name: str):
    return m11.load(name)


def starter_note(text: str) -> None:
    st.info(f"Starter display: {text} Build this section up to its required minimum.")


def recommendations() -> None:
    d = load("decisions")
    recs = (d or {}).get("recommendations", {})
    with st.container(border=True):
        st.subheader("Recommendations")
        if not any(recs.values()):
            st.caption("No recommendations yet: fill in `Recommendations` in DECISIONS.md, run "
                       "`python -m analysis`, then commit and push.")
        for k in m11.DATASETS:
            if any(recs.values()):
                st.markdown(f"**{m11.DATASET_LABEL[k]}:** {recs.get(k) or '_not written yet_'}")


# ---- 1. Beyond Accuracy ------------------------------------------------------
def test_confusion_table(c: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [[c["tp"], c["fn"]], [c["fp"], c["tn"]]],
        index=["Actual positive", "Actual negative"],
        columns=["Predicted positive", "Predicted negative"],
    )


def beyond_accuracy(res: dict, model: str) -> None:
    p, y = m11.preds(res, model)
    c = m11.confusion(p, y, 0.5)
    m = m11.metrics(c)
    always_no = m11.metrics(m11.confusion([0.0 for _ in y], y, 0.5))
    positive = {
        "readmission": "readmitted within 30 days of discharge",
        "diabetes": "doctor diagnosis of diabetes or HbA1c ≥ 6.5%",
    }[res["dataset"]]

    st.write(f"**Positive class:** patients with {positive}.")
    st.caption(f"Test patients only · {m['n']:,} patients")
    summary = st.columns(2)
    summary[0].metric("Test prevalence", m11.fmt_pct(m["prevalence"]))
    summary[1].metric("Test accuracy if we always predict no", m11.fmt_pct(always_no["accuracy"]))
    st.write("Accuracy counts all correct predictions, including patients without the outcome. "
             "Compare it with the always-no result before deciding whether a model helps find positive patients.")

    st.subheader(f"Test · {m11.MODEL_LABEL[model]} at threshold 0.50")
    st.write("A patient is predicted positive when the model probability is at least 0.50.")
    st.caption("Test confusion matrix · rows: actual outcome; columns: predicted outcome; positives first")
    st.table(test_confusion_table(c))
    st.write(f"**True positives ({c['tp']:,}):** patients with the outcome who were flagged.  "
             f"**False negatives ({c['fn']:,}):** patients with the outcome who were missed.  "
             f"**False positives ({c['fp']:,}):** patients without the outcome who were flagged.  "
             f"**True negatives ({c['tn']:,}):** patients without the outcome who were not flagged.")
    st.write(f"**Test sensitivity (recall / TPR): {m11.fmt_pct(m['sensitivity'])}.** "
             "Of the patients who had the outcome, this share was flagged.")
    st.write(f"**Test specificity: {m11.fmt_pct(m['specificity'])}.** "
             "Of the patients who did not have the outcome, this share was not flagged.")
    st.write(f"**Test precision: {m11.fmt_pct(m['precision'])}.** "
             "Of the patients who were flagged, this share had the outcome.")

    st.subheader("Test · compare all five models at threshold 0.50")
    rows = []
    for name in m11.MODELS:
        if name not in res["models"]:
            continue
        model_p, model_y = m11.preds(res, name)
        values = m11.metrics(m11.confusion(model_p, model_y, 0.5))
        rows.append({
            "Model": m11.MODEL_LABEL[name],
            "Accuracy": m11.fmt_pct(values["accuracy"]),
            "Sensitivity / recall / TPR": m11.fmt_pct(values["sensitivity"]),
            "Specificity": m11.fmt_pct(values["specificity"]),
            "FPR": m11.fmt_pct(values["fpr"]),
            "Precision": m11.fmt_pct(values["precision"]),
            "F1": m11.fmt_pct(values["f1"]),
        })
    st.caption("Test metrics · one row per model · undefined means the denominator is zero")
    st.table(pd.DataFrame(rows).set_index("Model"))

    st.caption("Test confusion matrices · rows: actual outcome; columns: predicted outcome; positives first")
    for name in m11.MODELS:
        if name not in res["models"]:
            continue
        model_p, model_y = m11.preds(res, name)
        counts = m11.confusion(model_p, model_y, 0.5)
        st.markdown(f"**Test · {m11.MODEL_LABEL[name]} · threshold 0.50**")
        st.table(test_confusion_table(counts))


# ---- 2. A Threshold for the Job ------------------------------------------------
def threshold(res: dict, model: str) -> None:
    starter_note("metrics at 0.50 only.")
    p, y = m11.preds(res, model)
    m = m11.metrics(m11.confusion(p, y, 0.5))
    st.caption(f"{TEST} · {m11.MODEL_LABEL[model]}, threshold 0.50")
    st.table(pd.DataFrame([{"Sensitivity": m11.fmt_pct(m["sensitivity"]), "Specificity": m11.fmt_pct(m["specificity"]),
                            "Precision": m11.fmt_pct(m["precision"]), "Flagged per 100": f"{m['flagged_per_100']:.1f}"}]))


# ---- 3. Training vs. Test ------------------------------------------------------
def train_vs_test(res: dict, model: str) -> None:
    starter_note("training and test AUROC of every model, in a table.")
    s = m11.auroc_summary(res)
    rows = [{"Model": m11.MODEL_LABEL[m], f"{TRAIN} AUROC": m11.fmt_num(v["train"]), f"{TEST} AUROC": m11.fmt_num(v["test"])}
            for m, v in s.items()]
    st.caption(f"{TRAIN} and {TEST} · AUROC on the rows each model learned from, and on rows it never saw")
    st.table(pd.DataFrame(rows).set_index("Model"))


RENDER = dict(zip(SECTIONS, [beyond_accuracy, threshold, train_vs_test]))

st.title("Judge the Model")
st.caption("Which classifier would you trust for this job, at what threshold, and with what limits?")
recommendations()
section = st.sidebar.radio("Section", SECTIONS)
key = st.sidebar.selectbox("Dataset", m11.DATASETS, format_func=m11.DATASET_LABEL.get)
res = load(key)
model = st.sidebar.selectbox("Model", m11.MODELS,
                             format_func=lambda m: m11.MODEL_LABEL[m] + ("" if m in res["models"] else " (not trained yet)"))
st.header(section)
st.caption(f"{m11.DATASET_LABEL[key]}: {res['train']['n']:,} training rows ({res['train']['n_pos']:,} positive), "
           f"{res['test']['n']:,} test rows ({res['test']['n_pos']:,} positive).")
if model in res["models"]:
    RENDER[section](res, model)
else:
    st.warning(f"{m11.MODEL_LABEL[model]} has no results yet: run `python -m analysis`, then commit and push `results/`.")
