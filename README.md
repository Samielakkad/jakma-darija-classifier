---
license: apache-2.0
language:
- ar
- fr
- en
tags:
- darija
- moroccan-arabic
- text-classification
- intent-classification
- low-resource
- arabic-dialects
- service-marketplace
library_name: transformers
base_model: xlm-roberta-base
pipeline_tag: text-classification
datasets:
- samielakkad1/jakma-darija-intents
metrics:
- accuracy
- f1
model-index:
- name: jakma-darija-classifier
  results:
  - task:
      type: text-classification
      name: Intent + City Classification (Darija)
    metrics:
    - type: accuracy
      value: 0.92
      name: 5-fold CV accuracy (initial baseline)
    - type: f1
      value: 0.89
      name: macro-F1
---

# AI + NLP · jakma-darija-classifier

> Pass-1 intent + city classifier for [jak.ma](https://jak.ma) — a production Moroccan-Arabic (Darija) service marketplace. Classifies natural-language queries in Darija / Arabic / French / English / mixed-script into one of 12 trade categories and one of 15 Moroccan cities, with confidence-gated routing.

**Author:** [Sami El Akkad](https://github.com/Samielakkad) · Tsinghua SIGS, MSc Artificial Intelligence
**Email:** sam25@mails.tsinghua.edu.cn
**Production system:** [jak.ma](https://jak.ma) (live since Apr 2026 · ~3,400 daily queries · p50 1.18s · $0.0008/query)
**Methodology:** [pm-frameworks-darija](https://github.com/Samielakkad/AI-Product-Management-Frameworks) · [jak-ma-eval-suite](https://github.com/Samielakkad/AI-LLM-Evaluation-JakMa)

---

## What this model does

Given a user query in any of: **Darija (Latin or Arabic script), Modern Standard Arabic, French, English, or code-switched mix**, this model predicts:

1. **Trade category** — one of 12 fixed classes (plumber, electrician, tiler, painter, carpenter, mason, mechanic, electronics, AC technician, gardener, cleaner, mover)
2. **City** — one of 15 Moroccan cities (Casablanca, Rabat, Salé, Tangier, Marrakesh, Agadir, Fes, Meknes, Oujda, Kenitra, Tétouan, Nador, Béni Mellal, El Jadida, Mohammedia)
3. **Confidence band** — `high` (route directly), `medium` (route with clarifying question), `low` (escalate to free-text fallback)

This is **Pass 1** of jak.ma's two-pass architecture. Pass 2 (generation + verifier) is a separate model.

## Example inputs

| Input | Expected output |
|---|---|
| `بغيت plombier f Casa daba` | `{trade: "plumber", city: "Casablanca", confidence: "high"}` |
| `kankhdem 3la lectricite f Tanger` | `{trade: "electrician", city: "Tangier", confidence: "high"}` |
| `Need a tiler in Rabat` | `{trade: "tiler", city: "Rabat", confidence: "high"}` |
| `chi wahed yeqdar yssawb chi haja f darna` | `{trade: null, city: null, confidence: "low"}` (handyman ambiguous) |

## Architecture

- **Base model:** `xlm-roberta-base` (frozen) — chosen for native Arabic + Latin script support, public license, mid-weight (~278M params)
- **Heads:**
  - Linear head for trade classification (12 classes)
  - Linear head for city classification (15 classes)
  - Confidence head (3 levels) trained on softmax temperature + verifier disagreement signals
- **Tokenizer:** XLM-R sentencepiece, extended with 800 Darija-frequent tokens from a 50K-line jak.ma chat corpus
- **Training:** LoRA-tuned (r=16, alpha=32) on `samielakkad1/jakma-darija-intents` — see Dataset section
- **Inference:** ~30ms p50 on T4, runs CPU-friendly via ONNX export

## How it's used in production

```python
from handler import EndpointHandler

clf = EndpointHandler("/path/to/model-artifacts")
out = clf({"inputs": "بغيت plombier f Casa daba"})
# → [{"trade": "plumber", "city": "Casablanca", "confidence": "high", ...}]
```

In jak.ma production this is wrapped with a verifier (see [jak-ma-eval-suite/VERIFIER_SPEC.md](https://github.com/Samielakkad/AI-LLM-Evaluation-JakMa/blob/main/VERIFIER_SPEC.md)) — if `score < 0.6`, the system asks a clarifying question instead of routing.

### Inference artifact contract

The model directory passed to `EndpointHandler` must contain:

- `config.json`, including `hidden_size`, each `num_labels_*` count, and contiguous `id2label_*` mappings.
- `heads.pt`, saved as the state dict of a `torch.nn.ModuleDict` with `trade`, `city`, and `confidence` linear heads.
- The tokenizer and XLM-R encoder artifacts expected by `AutoTokenizer.from_pretrained` and `AutoModel.from_pretrained`.

The handler validates label IDs, label counts, encoder width, checkpoint keys, and every checkpoint tensor shape before serving traffic. `heads.pt` is loaded on CPU with restricted tensor-only deserialization and is moved to the inference device only after validation. A missing or incompatible artifact is a startup error; the handler never substitutes randomly initialized heads.

`inputs` accepts either one non-empty string or a non-empty list of non-empty strings. Batch requests use one padded encoder pass and return one prediction object per input, in the same order. Invalid input types and blank strings raise an explicit `TypeError` or `ValueError`.

## Development checks

The unit tests exercise artifact validation and batched inference without downloading model weights:

```bash
python -m unittest discover -s tests -v
python -m compileall -q handler.py tests
```

## Dataset

Training data: a curated subset of jak.ma's first 50,000 production queries (anonymized, Oct 2025 – May 2026), human-labeled to the trade × city schema, with a 200-worker field survey used to anchor the city-trade plausibility table.

Splits:
- Train: 40,000 queries
- Val: 5,000 queries
- Test (held-out): 5,000 queries from May 2026 (most recent slice, used to catch concept drift)

Label distribution is **deliberately imbalanced** to match production traffic — Casablanca and Rabat dominate (~55% combined), plumbing + electrical dominate trades (~40% combined). Macro-F1 is the headline metric, not accuracy.

The dataset is hosted separately at [`samielakkad1/jakma-darija-intents`](https://huggingface.co/datasets/samielakkad1/jakma-darija-intents) (CC-BY-4.0, with PII removed).

## Evaluation methodology

This classifier was developed using the **5-dimension evaluation rubric** from the Baidu ERNIE Mentor Program, adapted for the marketplace domain. See:

- [ernie-evaluation-notes](https://github.com/Samielakkad/AI-LLM-Evaluation-Baidu-ERNIE) — the source methodology
- [pm-frameworks-darija/evaluation-rubric.md](https://github.com/Samielakkad/AI-Product-Management-Frameworks/blob/main/evaluation-rubric.md) — the marketplace adaptation

**Headline metric (held-out test set):**
- Trade classification: 92.4% accuracy, 89.1% macro-F1
- City classification: 94.7% accuracy, 91.8% macro-F1
- Joint (both correct): 87.3%
- Confidence calibration (ECE): 0.041

**Eval slices:**
- Code-switched queries (43% of corpus): 88.1% joint accuracy
- Pure Darija (Arabic script): 86.4%
- Pure Darija (Latin script / Arabizi): 89.2%
- Pure French: 91.7%
- Pure English: 93.4%

## Limitations + honest disclosures

1. **Tangier and Agadir are weak** — fewer than 200 training examples each, F1 drops to ~0.78 on these cities. On roadmap to fix via active sampling.
2. **Handyman / "shi haja" queries** — the model deliberately routes ambiguous queries to `confidence: low` rather than guessing. This is by design (it pairs with a verifier downstream) but means raw accuracy on the "general repair" intent class looks artificially low.
3. **Salé / Rabat adjacency** — these two cities are 5km apart across a river and the model can flip between them on commuter queries. An adjacency post-processor handles this in production (see [jak-ma-eval-suite](https://github.com/Samielakkad/AI-LLM-Evaluation-JakMa)).
4. **Concept drift** — pricing-sensitive trade labels (e.g. AC technician) drift seasonally. Model is retrained quarterly. Last retrain: May 2026.
5. **Not a chat model** — this is *classification only*. The Darija chat experience uses this as Pass 1, with a separate Pass 2 generator + verifier.

## Bias + ethics

- The model is trained on real Moroccan user queries — it inherits production usage biases (urban-skewed, working-age-skewed, more Casablanca/Rabat).
- **No phone numbers, names, or location coordinates** are in the training data (PII stripped at ingestion).
- The model is intentionally NOT used to filter or rank workers — it only handles intent extraction. Worker selection is deterministic + verifier-gated downstream.

## Citation

```bibtex
@misc{elakkad2026jakma,
  author       = {El Akkad, Sami},
  title        = {jakma-darija-classifier: Pass-1 intent and city classifier for a Moroccan-Arabic service marketplace},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/samielakkad1/jakma-darija-classifier}}
}
```

## Related artifacts

- 🤗 [jakma-darija-chat Space](https://huggingface.co/spaces/samielakkad1/jakma-darija-chat) — interactive demo wrapping this classifier + Pass 2 generator
- 📐 [pm-frameworks-darija](https://github.com/Samielakkad/AI-Product-Management-Frameworks) — the 5-file PM/eval methodology
- 📊 [jak-ma-eval-suite](https://github.com/Samielakkad/AI-LLM-Evaluation-JakMa) — verifier spec + benchmark prompts
- 📝 [jak-ma-case-study](https://github.com/Samielakkad/AI-Product-JakMa-Case-Study) — production case study
- 🎓 [Methodology origin · ernie-evaluation-notes](https://github.com/Samielakkad/AI-LLM-Evaluation-Baidu-ERNIE) — Baidu ERNIE Mentor Program methodology this classifier inherits from
- 🌐 [Portfolio](https://samielakkad.github.io)

## License

Apache 2.0 (model) · CC-BY-4.0 (model card text) · Underlying dataset CC-BY-4.0 with PII removed.

---

*Built with the 5-criteria eval rubric originally designed at DOPEDCLUB (Casablanca), formalised during the Baidu ERNIE Mentor Program (Shenzhen), now running in production at jak.ma (Morocco). Same disagreement-based verifier reasoning applied at runtime.*


---

**License — All rights reserved.** This repository is shared for review only. Please **contact me before using any part of it** for any purpose. See [LICENSE](LICENSE).
