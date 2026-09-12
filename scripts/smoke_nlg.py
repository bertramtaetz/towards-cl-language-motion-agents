"""Run real NLG scorers and write strict JSON; no model download is allowed."""
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MOTION_NLG_ROOT", str(ROOT / "pretrained/nlg"))


def main():
    from nlgeval import NLGEval
    evaluator = NLGEval(metrics_to_omit=["METEOR", "EmbeddingAverageCosineSimilarity",
                       "SkipThoughtCS", "VectorExtremaCosineSimilarity", "GreedyMatchingScore"])
    refs = [["a person walks forward and then sits down", "a person raises both arms and jumps"]]
    scores = evaluator.compute_metrics(ref_list=refs, hyp_list=refs[0])
    for key in ["Bleu_1", "Bleu_4", "ROUGE_L", "CIDEr", "SPICE"]:
        if key not in scores or not math.isfinite(scores[key]):
            raise RuntimeError(f"Missing or invalid required metric: {key}: {scores}")
    out = ROOT / "outputs/smoke/nlg_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=2, allow_nan=False) + "\n")
    print(out)
    print(scores)


if __name__ == "__main__":
    main()