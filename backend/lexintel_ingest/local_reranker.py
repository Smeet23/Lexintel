"""Score query and passage pairs with a local cross-encoder."""

from itertools import batched


class LocalReranker:
    def __init__(self, model_path, *, device="cpu"):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True).to(device).eval()
        self.device = device

    def score(self, question, candidates):
        import torch

        scores = []
        with torch.inference_mode():
            for batch in batched(candidates, 16):
                inputs = self.tokenizer(
                    [question] * len(batch), [row["passage"]["search_text"] for row in batch],
                    max_length=512, padding=True, truncation="only_second", return_tensors="pt",
                ).to(self.device)
                scores.extend(self.model(**inputs).logits.reshape(-1).tolist())
        return scores

    def rank(self, question, candidates):
        scores = self.score(question, candidates)
        return [row for row, _ in sorted(zip(candidates, scores, strict=True), key=lambda pair: -pair[1])]
