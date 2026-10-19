# Choosing the open-weight LLM for the State Card

**Recommendation:** use **Qwen3.5 9B** (Apache-2.0, `ollama pull qwen3.5`) with thinking **off** as the
default narrator. Use **Gemma 4 E4B** (Apache-2.0, `ollama pull gemma4:e4b`) where speed or GPU memory is
tight. Both passed every check on all four demo patients. Both run on a single 8 GB laptop GPU, at about
20 s and 10 s per card respectively.

## What the LLM has to do

It writes only the `Narrative` section of the card from a cue digest, under JSON-schema-constrained decoding.
It must:
- return valid JSON;
- stay grounded: cite only real check-in ids, stressor domains and numbers, and get "N of M check-ins" counts right;
- avoid diagnoses and disorder labels;
- avoid treatment advice;
- not comment on risk, which the safety layer handles.

The schema is built per request. Evidence ids and stressor domains are enums of what exists in the data,
so the decoder cannot produce an unknown id at all.

## Candidates (October 2026)

| Model | Licence | Size (Ollama Q4) | Why considered |
|---|---|---|---|
| Qwen3.5 9B | Apache-2.0 | 6.6 GB | Strongest open model under 10B on independent indexes; released March 2026 |
| Qwen3.5 4B | Apache-2.0 | 3.3 GB | Same family, fits 4 GB GPUs |
| Gemma 4 E4B | Apache-2.0 (Gemma 4 dropped the custom Gemma licence) | 6.6 GB | Built for function calling and structured JSON on constrained hardware |
| Mistral 7B | Apache-2.0 | 4.4 GB | Older baseline already installed |
| Sarvam-30B / 105B | Apache-2.0 (March 2026) | 30B MoE, needs a server GPU | Built for 22 Indian languages; **not tested here** (too large for the 8 GB test machine) |

Llama models were skipped because of the community licence. NRC-style and other non-commercial
resources were already ruled out for the product.

## Benchmark (`python src/run_phase1.py bench`)

Four fictional demo patients cover English (improving), Hinglish (worsening, ideation), Devanagari Hindi
(mild) and English crisis. Each model gets one attempt per patient (no retry), via Ollama 0.40.6, on an RTX
4060 Laptop (8 GB) GPU. Full table: `outputs/phase1/llm_benchmark.md`.

| Model | Schema-valid | Passes all guardrails | Mean latency | Notes |
|---|---|---|---|---|
| **Qwen3.5 9B, think off** | 4/4 | **4/4** | 20 s | Accurate counts and stressors, including Hinglish ("relationship conflict", "EMI") |
| **Gemma 4 E4B, think off** | 4/4 | **4/4** | 10 s | Most concise (~47 words); once called a single GAD-7 "stable" |
| Qwen3.5 4B, think off | 4/4 | 2/4 | 12 s | Invents counts ("2 of 7 check-ins" when the data says 1) |
| Mistral 7B | 4/4 | 2/4 | 21 s | Uses diagnostic labels ("mild depression"), often over the word limit |
| Qwen3.5 9B, think on | 1/4 | 1/4 | 241 s | Usually returns empty content with a schema |
| Qwen3.5 4B, think on | 2/4 | 2/4 | 120 s | Same problem |

The think-on rows come from an earlier run with fewer guardrails. Their failures were empty or truncated
JSON, which no guardrail can fix.

### What the benchmark taught us (and what changed in the code)

1. **Thinking mode and schema-constrained output don't mix in Ollama.** Content often came back empty and
   was 5–10× slower. This matches an open Ollama issue on Qwen 3.5 / 3.6 JSON-schema adherence. Keep
   `think=False` (the default in `OllamaBackend`).
2. **"Schema-valid" is not "correct".** Every model produced valid JSON almost every time. The real
   failures were grounding problems, each now caught by a guardrail and fixed by the one retry:
   - Gemma cited JSON paths (`questionnaires.PHQ-9.history[0]`) as evidence ids. **Fix:** a per-request
     enum schema for ids.
   - Mistral and Qwen 4B copied the prompt's example "4 of 8 check-ins" into real summaries. **Fix:** no
     concrete numbers in the prompt, and an "N of M check-ins" check against the real counts (including
     number words).
   - VADER rates "everyone would be better off without me" as positive, and Qwen then wrote "mood
     improved". **Fix:** lexicon sentiment was removed from the LLM's input.
   - One PHQ-9 administration was being called "stable". **Fix:** a new `baseline_only` direction.
3. **Retry plus fallback make the card dependable.** In the final demo run (Qwen3.5 9B, full neural
   pipeline), 3 cards passed on the first attempt and 1 needed the retry. If both attempts fail, the
   rule-based narrative is used and the card still ships.

## Deployment path

- **Pilot / on-premise:** Ollama with `qwen3.5` (or `gemma4:e4b`) on the hospital's own GPU box. No patient
  text leaves the hospital.
- **Scale-up, Indian cloud region (Azure India / AWS Mumbai):** serve the same weights with vLLM or SGLang
  structured outputs (xgrammar) through `OpenAICompatBackend` (`--backend openai --base-url ...`). The
  schema, guardrails and fallback are unchanged.
- **Phase 2 evaluation:** run `bench` on de-identified pilot cards with clinician ratings. Add Sarvam-30B
  and larger Qwen models once a server GPU is available. Hindi-language narratives (rather than English
  summaries of Hindi check-ins) are where an Indic-specialised model should earn its place.

## Caveats

- Four fictional patients is a small benchmark. It shows which failure modes exist and that the
  guardrails catch them; it does not establish clinical quality. Clinician review of real summaries is the
  real test.
- Latency is for a laptop GPU with the 9B model partly offloaded to CPU. A dedicated 16–24 GB GPU will be
  several times faster.

## Sources

- [Qwen3.5 small models: sizes, Apache-2.0, benchmarks (Artificial Analysis)](https://artificialanalysis.ai/articles/qwen3-5-small-models)
- [Qwen3.5 small series release (Gigazine)](https://gigazine.net/gsc_news/en/20260303-qwen-3-5-small)
- [Gemma 4 open model and Apache-2.0 licence](https://www.buildfastwithai.com/blogs/google-gemma-4-open-model)
- [Sarvam 30B and 105B under Apache-2.0](https://www.opensourceforu.com/2026/03/sarvam-releases-30b-and-105b-llms-under-apache-2-0/), [Sarvam AI (Wikipedia)](https://en.wikipedia.org/wiki/Sarvam_AI)
- [Ollama structured outputs](https://registry.ollama.ai/blog/structured-outputs), [structured output with local LLMs, incl. the Qwen 3.5 adherence issue](https://insiderllm.com/guides/structured-output-local-llms/)
- [Open-source LLM landscape 2026](https://mentiko.com/blog/open-source-llm-landscape-2026)
