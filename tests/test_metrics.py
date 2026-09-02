from eval.metrics import (
    char_overlap_ratio,
    human_review_rate,
    match_spans,
    precision_recall_f1,
    verification_catch_rate,
)


def test_char_overlap_ratio_full_overlap():
    pred = {"start_char": 10, "end_char": 20}
    gt = {"start_char": 10, "end_char": 20}
    assert char_overlap_ratio(pred, gt) == 1.0


def test_char_overlap_ratio_no_overlap():
    pred = {"start_char": 0, "end_char": 5}
    gt = {"start_char": 10, "end_char": 20}
    assert char_overlap_ratio(pred, gt) == 0.0


def test_match_spans_counts_tp_fp_fn():
    predicted = [
        {"text": "Rajesh Mehta", "start_char": 0, "end_char": 12},
        {"text": "extra false positive", "start_char": 50, "end_char": 71},
    ]
    ground_truth = [
        {"text": "Rajesh Mehta", "start_char": 0, "end_char": 12},
        {"text": "missed one", "start_char": 100, "end_char": 110},
    ]
    result = match_spans(predicted, ground_truth)
    assert result == {"tp": 1, "fp": 1, "fn": 1}


def test_precision_recall_f1_basic():
    metrics = precision_recall_f1(tp=8, fp=2, fn=2)
    assert metrics["precision"] == 0.8
    assert metrics["recall"] == 0.8
    assert round(metrics["f1"], 3) == 0.8


def test_precision_recall_f1_handles_zero_division():
    metrics = precision_recall_f1(tp=0, fp=0, fn=0)
    assert metrics == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_verification_catch_rate():
    results = [
        {"simulated_failure": True, "verifier_passed": False},  # caught
        {"simulated_failure": True, "verifier_passed": True},  # missed
        {"simulated_failure": False, "verifier_passed": True},  # correct
        {"simulated_failure": False, "verifier_passed": False},  # false alarm
    ]
    metrics = verification_catch_rate(results)
    assert metrics["verification_catch_rate"] == 0.5
    assert metrics["false_alarm_rate"] == 0.5


def test_human_review_rate():
    docs = [
        [{"confidence": 0.9}, {"confidence": 0.5}],  # has a low-confidence flag
        [{"confidence": 0.95}],  # all confident
        [],  # no flags at all
    ]
    assert human_review_rate(docs, confidence_threshold=0.7) == 1 / 3
