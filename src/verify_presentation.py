"""Check every participant/reliability display cell against probability exports."""

from pathlib import Path
from collections import defaultdict
import json
import math
from verify import rows, selective, close

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
selection = json.loads((R / "selection.json").read_bytes())
groups = defaultdict(list)
people = defaultdict(list)
for row in rows(R / "test_predictions.csv.gz"):
    groups[(row["model"], row["condition"])].append(row)
    people[(row["model"], row["participant"], row["condition"])].append(row)
participant_count = 0
for display in rows(R / "participant_metrics.csv"):
    block = people[(display["model"], display["participant"], display["condition"])]
    expected = selective(block, selection["thresholds"].get(display["model"], 0.0))
    for key, value in expected.items():
        if value is None:
            assert display[key] == ""
        else:
            close(display[key], value)
    correct = sum(
        max(range(6), key=lambda i: float(row[f"p{i}"])) == int(row["label"])
        for row in block
    )
    close(display["accuracy"], correct / len(block))
    participant_count += 1
bin_count = 0
for display in rows(R / "reliability.csv"):
    bucket = int(round(float(display["lower"]) * 10))
    block = groups[(display["model"], display["condition"])]
    matching = []
    for row in block:
        p = [float(row[f"p{i}"]) for i in range(6)]
        confidence = max(p)
        if min(int(confidence * 10), 9) == bucket:
            matching.append(
                (confidence, max(range(6), key=lambda i: p[i]) == int(row["label"]))
            )
    assert int(display["n"]) == len(matching), (display, len(matching))
    if matching:
        close(display["confidence"], math.fsum(v[0] for v in matching) / len(matching))
        close(display["accuracy"], sum(v[1] for v in matching) / len(matching))
    else:
        assert display["confidence"] == display["accuracy"] == ""
    bin_count += 1
assert participant_count == 81 and bin_count == 90
receipt = {
    "status": "PASS",
    "participant_display_rows": participant_count,
    "reliability_bins": bin_count,
    "independent_bin_definition": "Integer floor of confidence*10, capped at9; probability exports only",
    "scope": "All display cells reconstructed; separate actual PNG/PDF visual review required",
}
(R / "INDEPENDENT_PRESENTATION_CHECK.json").write_text(
    json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(receipt), flush=True)
