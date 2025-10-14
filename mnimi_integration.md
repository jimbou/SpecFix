# Using Mnimi LLM Cache with SpecFix

Mnimi adds deterministic yet flexible caching semantics to SpecFix without altering the tool's agentic workflow. This guide describes how to route SpecFix's LLM traffic through Mnimi so you can mix independent retries with repeatable, disk-backed replays.

## Why combine Mnimi with SpecFix?

SpecFix issues dozens of completions while it diagnoses ambiguous requirements, regenerates programs, and validates repairs. Those calls are orchestrated through a single `Model` helper, meaning one integration point can deliver three practical benefits:

* **Reproducible experiments** — Persistent caching replays byte-identical completions across repeated CLI runs, eliminating drift when you change other hyper-parameters.
* **Cheaper retries** — Repeatable caching prevents redundant spending when you inspect the same attempt multiple times, while independence can still be requested exactly where the algorithm needs fresh draws.
* **Deterministic debugging** — Once the cache is primed, you can debug downstream Python logic without re-querying the LLM.

## How SpecFix currently calls the LLM

The evaluator constructs a `Model` instance in its initializer and forwards every completion request through `get_response` or `get_response_sample` during detection, sampling, and repair.【F:evaluator.py†L12-L177】 The `Model` wrapper itself is a thin OpenAI-compatible client that returns either single completions or batched samples depending on the call site.【F:model.py†L5-L94】 Because this abstraction already owns the API key and batching logic, swapping it for a Mnimi-backed equivalent gives you control over every LLM touchpoint in the pipeline.

## Integrating Mnimi step by step

1. **Instantiate a Mnimi provider** using the same model name and `LLM_API_KEY` that SpecFix already expects. Any of Mnimi's HTTP adapters (AI302, CloseAI, FireworksAI, XMCP, etc.) can stand in for the current OpenAI client.
2. **Layer persistence first** by wrapping the provider with `Persistent` and choosing a cache directory (e.g., `~/.mnimi/specfix`). This records all completions on disk so later runs can replay them.
3. **Thread the wrapper into SpecFix** by letting `SpecFixAccuracyEvaluator` accept an optional model argument. When provided, skip creating the default `Model` and reuse the injected Mnimi stack instead.【F:evaluator.py†L12-L18】 Existing code paths continue to call `get_response` / `get_response_sample`, so no behavioural changes occur elsewhere.
4. **Add convenience hooks at the CLI layer** (for example, in `main.py`) to opt into Mnimi by pointing to the cache directory. This keeps legacy usage untouched while enabling caching with a single flag.

## Stage-specific caching strategy

Once Mnimi powers the evaluator, you can decide where independence versus repeatability matters most:

1. **Detection phase** — `specfix_detect` first launches `generate_tests`, retrying up to ten times to extract a satisfactory suite.【F:evaluator.py†L161-L177】【F:evaluator.py†L81-L103】 SpecFix now calls `get_response(..., cache_mode="independent")`, which advances the Mnimi iterator on every retry while still persisting the eventually accepted tests for future runs.【F:evaluator.py†L81-L105】【F:model.py†L55-L148】
2. **Initial program sampling** — After tests are locked in, `generate_programs` issues either batched samples or multiple parallel requests, ultimately funnelling through `get_response_sample` and `generate_program` for fallbacks.【F:evaluator.py†L32-L80】【F:evaluator.py†L135-L159】 Both paths request `cache_mode="independent"` so new programs appear whenever the evaluator widens the search space, yet the persistent layer still replays the same batches once the cache is warm.【F:evaluator.py†L32-L80】【F:model.py†L101-L140】
3. **Repair loop** — `specfix_repair` iterates up to three times, alternating between program repair, contrastive inference, and a fresh sampling pass before evaluating consistency gains.【F:evaluator.py†L179-L222】 The repair routines choose `cache_mode="repeatable_attempt"`, which nests a repeatable cache on top of the shared independent stream so every attempt receives a fresh completion while repeated reads within an attempt stay deterministic.【F:evaluator.py†L179-L222】【F:model.py†L55-L148】

These layers compose naturally as `Persistent → Independent → Repeatable`, giving you deterministic behaviour within a single attempt, independence across retries, and full reproducibility between runs.【F:model.py†L55-L148】

## Operational checklist

* Decide on a cache directory and expose it through a CLI flag or environment variable.
* On evaluator construction, detect that flag and build the Mnimi provider stack (`Persistent`, optionally nested with `Independent` / `Repeatable`) instead of the default `Model`.
* For debugging or strict replication, enable Mnimi's `replication=True` mode so cache misses raise `ReplicationCacheMiss`, guaranteeing the run never falls back to live queries.

Following this checklist ensures SpecFix remains statistically sound—fresh generations appear where the algorithm depends on them—while you capture the reproducibility and cost savings that Mnimi provides.