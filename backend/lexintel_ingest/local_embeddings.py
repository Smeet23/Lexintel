"""Local E5 encoding for offline retrieval experiments."""

from itertools import batched


class LocalEmbeddings:
    def __init__(self, model_path):
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True).eval()

    def encode(self, texts, kind):
        import torch
        from torch.nn.functional import normalize

        vectors = []
        with torch.inference_mode():
            for batch in batched(texts, 16):
                inputs = self.tokenizer(
                    [f"{kind}: {text}" for text in batch],
                    max_length=512, padding=True, truncation=True, return_tensors="pt",
                )
                states = self.model(**inputs).last_hidden_state
                mask = inputs["attention_mask"][..., None].bool()
                mean = states.masked_fill(~mask, 0.0).sum(dim=1) / mask.sum(dim=1)
                vectors.append(normalize(mean, dim=1))
        return torch.cat(vectors)
