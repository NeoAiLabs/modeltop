# ModelTop

ModelTop is a terminal application for monitoring and benchmarking local OpenAI-compatible LLM servers. It is designed as a fast, keyboard-driven dashboard for engineers running local inference stacks.

## Status

Version `0.1.0` provides a live Textual dashboard, inline Chat playground, sequential single-request Speed Test, simultaneous streaming Concurrency benchmark, progressive Context Length benchmark, native Tool Calling benchmark, and sequential Drafter speculative-decoding benchmark for one configured generic OpenAI-compatible server and the machine running ModelTop. It discovers and selects models from `GET /v1/models`, streams responses from `POST /v1/chat/completions`, reports live, percentile, scaling, retrieval, tool-use, speculative draft/accept, and local-hardware metrics, retains bounded session results, exports allowlisted Speed Test JSON, and records sanitized file logs.

## Prerequisites

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- A terminal supported by Textual

## Setup

Install the project and its development dependencies:

```bash
uv sync
```

The lockfile and virtual environment are managed by uv.

## Launch

Run the installed console command:

```bash
uv run mtop
```

Or launch the package as a Python module:

```bash
uv run python -m modeltop
```

Press `Tab` to cycle focus. In the sidebar, use the arrow keys and `Enter` to open Overview, Chat, Speed Test, Concurrency, Context Length, Tool Calling, Drafter, Results, or Settings. In Chat, `Enter` or `Ctrl+Enter` sends, `Shift+Enter` or `Alt+Enter` inserts a newline, `Esc` cancels, `Ctrl+K` clears, and `Ctrl+G` toggles settings. In Speed Test, `R` reruns terminal detail, `E` exports JSON, and `C` copies a summary. Concurrency, Context Length, Tool Calling, and Drafter use the Run button or plain `R`; `Esc` cancels active work or returns terminal detail to configuration, and terminal `E` edits the exact configuration. Context Length asks for explicit confirmation before reserving large runs; Tool Calling asks before the default 69-scenario Full suite. `Ctrl+L` refreshes monitoring when no benchmark owns endpoint traffic, `?` opens shortcut help, `Q` quits outside text inputs, and `Ctrl+Q` always quits.

## Configuration

ModelTop loads the first authoritative source in this order:

1. The nonblank `MODELTOP_CONFIG` environment variable. Relative paths resolve from the current directory.
2. `~/.config/modeltop/config.yaml`.
3. `config/modeltop.yaml` under the current directory.
4. Built-in defaults matching the local vLLM example below.

An explicit environment path and the first existing implicit file are authoritative: a missing, unreadable, or invalid selected file produces a configuration error instead of falling through. `config/modeltop.yaml` and `config/modeltop.example.yaml` are source-checkout assets, not wheel runtime data. Built-in defaults keep installed execution functional when no file exists.

The tracked checkout configuration selects `local-8888`. Built-in defaults and `config/modeltop.example.yaml` select port 8000:

```yaml
application:
  refresh_interval_seconds: 5
  request_timeout_seconds: 5
  default_server: local-vllm  # local-8888 in config/modeltop.yaml
hardware:
  enabled: true
  refresh_interval_seconds: 2
  preferred_provider: auto
benchmarks:
  concurrency:
    default_levels: [1, 2, 4, 8]
    requests_per_level: 16
    warmup_requests: 2
    max_tokens: 256
    temperature: 0.0
    top_p: 1.0
    request_timeout_seconds: 120.0
    delay_between_levels_seconds: 3.0
    maximum_concurrency: 128
  context:
    default_mode: sweep
    default_lengths: [1024, 4096, 8192, 16384, 32768]
    context_unit: tokens
    repetitions_per_length: 3
    warmup_requests: 1
    content_source: synthetic
    base_text: null
    random_seed: 42
    maximum_output_tokens: 128
    temperature: 0.0
    top_p: 1.0
    seed: 42
    request_timeout_seconds: 300.0
    delay_between_lengths_seconds: 3.0
    maximum_context_test_tokens: 262144
    warning_threshold_tokens: 65536
    prompt_target_tolerance_percent: 1.0
    hardware_sample_interval_seconds: 0.5
    estimated_input_rate_enabled: true
    reuse_prompt: true
    unique_prompt_suffix_per_run: false
    early_stop_enabled: true
    continue_after_timeout: true
    probe:
      start_tokens: 4096
      maximum_tokens: 131072
      resolution_tokens: 1024
    retrieval:
      enabled: false
      positions: [beginning, middle, end]
      key: null
      maximum_output_tokens: 32
      case_insensitive_match: false
      containment_match: false
      truncation_detection: true
      regenerate_per_run: true
  tool_calling:
    default_suite: full
    request_timeout_seconds: 120.0
servers:
  - id: local-vllm
    name: Local vLLM
    base_url: http://127.0.0.1:8000/v1
    api_key: EMPTY
    backend_hint: vllm
    default_model: null
  - id: local-8888
    name: Local 8888
    base_url: http://127.0.0.1:8888/v1
    api_key: EMPTY
    backend_hint: null
    default_model: null
```

To monitor port 8888 instead, set `application.default_server: local-8888`. ModelTop validates every configured server but monitors one selected server at a time.

A generic OpenAI-compatible server can use a custom API prefix:

```yaml
application:
  refresh_interval_seconds: 10
  request_timeout_seconds: 3
  default_server: generic
servers:
  - id: generic
    name: Generic Server
    base_url: https://llm.example.com/openai/v1
    api_key: replace-with-a-real-key
    backend_hint: null
    default_model: organization/model-name
```

The URL is treated as an API root: ModelTop preserves a terminal `/v1` or appends `/v1`, then uses relative `models` and `chat/completions` paths so custom prefixes remain intact. `backend_hint` is display-only and never selects an adapter. Configurations may define multiple servers, but the dashboard monitors only `application.default_server`, or the first server when that field is `null`.

Hardware metrics always describe the local machine running ModelTop; configured server URLs are never used for hardware collection. `preferred_provider: auto` initializes NVML once and falls back to non-blocking `nvidia-smi` when NVML is unavailable. Use `nvml` or `nvidia-smi` to force one provider. Either `enabled: false` or `preferred_provider: disabled` disables hardware monitoring. Files that omit `hardware` or `benchmarks` retain the defaults shown above. Concurrency levels must be positive, unique, sorted after runtime form validation, and no greater than `maximum_concurrency`; measured and warm-up counts each permit at most 1,000 requests.

## Chat playground

Chat uses the currently selected discovered model and keeps the header, footer, Overview, server refresh, and hardware collection mounted. Switching views or refreshing monitoring does not stop an active generation. A model selection made during generation applies to the next request; the active request retains its captured server, model, settings, and context.

Each request sends the generic OpenAI-compatible payload to `POST /v1/chat/completions`: the model ID, ordered `{role, content}` messages, temperature, top-p, maximum tokens, optional seed, `stream: true`, and optional `stream_options.include_usage`. The trimmed system prompt is sent at most once, followed by all prior user/assistant turns and the new user prompt. Conversation context is resent in full and is not automatically trimmed.

The default settings are temperature `0.7`, top-p `0.95`, maximum tokens `1024`, no seed, and streaming enabled. `Ctrl+G` opens runtime controls for these values, the system prompt, and transcript visibility of that prompt. Invalid changes are rejected without replacing prior settings. Settings cannot change and the conversation cannot be cleared while a generation is active.

Assistant Markdown is displayed incrementally. `Esc` closes the HTTP response stream and keeps a nonempty partial assistant response as an ordered conversation turn; cancellation is best-effort server-side, so a remote backend may continue work after the client disconnects. A subsequent request can start immediately after cancellation. Unsupported `stream_options` are retried once without usage options. A backend that explicitly rejects streaming is retried once in non-stream mode, and a 2xx JSON completion that ignored streaming is consumed without a duplicate request. The UI labels either case as a non-stream fallback.

Generation metrics use monotonic timestamps:

- TTFT is the time from request start to the first nonempty streamed content delta.
- Generation duration is completion time minus first-token time; total duration includes TTFT.
- Output tokens/second is completion tokens divided by positive generation duration. Inter-token latency is its reciprocal in milliseconds.
- Server usage wins field by field. Without server usage, an injected exact tokenizer may supply counts. Otherwise ModelTop displays `~` estimates using `ceil(non-whitespace characters / 4)`.
- Counts and derived throughput based on estimated counts retain the `~` marker. Missing, zero-duration, empty-output, and non-stream timing values display `--`, not fabricated zero or infinity.

Conversations and settings are in memory only and are not persisted or exported. Without server usage or an injected tokenizer, provider chat-template overhead is unknown and is not included in approximate prompt counts.

## Single-request Speed Test

Speed Test measures one request at a time against the currently selected discovered model. Start validates and freezes the displayed prompt, preset, generation controls, selected server/model, backend label, and pre-run hardware snapshot. Warm-ups run first and are retained for diagnostics but excluded from aggregates; measured requests then run strictly sequentially with no overlap. Each request uses a fresh one-message conversation and the shared Chat transport, parser, fallback, timeout, token-counting, and monotonic metric collector. Speed Test never appends to Chat history.

The built-in presets are Quick (`1` warm-up, `3` measured, `128` maximum tokens), Standard (`1`, `5`, `256`), and Long (`1`, `3`, `1024`). All use temperature `0`, top-p `1`, seed `42`, a `300` second per-request timeout, and stop-on-error. Custom accepts a nonblank prompt, `0`–`20` warm-ups, `1`–`100` measured requests, `1`–`32768` maximum tokens, temperature `0`–`2`, top-p greater than `0` and at most `1`, an optional integer seed, a positive timeout, and an optional continue-on-error policy. Editing any numeric, prompt, or policy field selects Custom; validation occurs once on Start and an invalid draft causes no network request.

Live progress identifies warm-up versus measured phase, request number, TTFT, token counts, throughput, elapsed duration, and a bounded response preview. `Esc` closes the active HTTP stream and retains a nonempty partial request record; cancellation remains best-effort server-side. By default, the first request failure stops the sequence and retains completed and partial diagnostics. Continue-on-error attempts the remaining requests and produces `COMPLETED WITH ERRORS`; if every measured request fails, aggregate counts are zero and aggregate values remain unavailable rather than becoming fabricated zeros. A missing or unavailable local GPU snapshot never blocks execution.

Terminal results retain the full frozen configuration, sanitized server/model metadata, before/after hardware snapshots, warm-up and measured request rows, exact-versus-estimated token flags, and measured-request mean, median, minimum, maximum, nearest-rank p95, and sample standard deviation. Results remain available in newest-first Results history until ModelTop exits. `R` reserves a new run from the saved configuration; it never mutates the original result.

`E` atomically exports an allowlisted schema-versioned JSON document to `~/.local/share/modeltop/results` by default. Filenames contain UTC time, sanitized server/model components, and the run ID; collisions receive deterministic suffixes. The export contains the full validated configuration, including the benchmark prompt, so treat it as potentially sensitive. It never includes API keys, authorization headers, endpoint query/fragment data, or generated response text. `C` requests OSC 52 clipboard delivery for a compact plain-text summary containing no prompt or response text; terminal support varies, so JSON export is the reliable fallback.

## Concurrency benchmark

Concurrency sends the exact same fixed prompt and generation settings through the shared streaming `GenerationService`, without touching Chat history. Fixed mode has one concurrency level: **concurrency is the maximum number of simultaneous requests, not the total request count**. Sweep mode sorts unique levels ascending, completes one level before starting the next, and never overlaps levels. Each level runs its configured warm-ups first through the same bounded worker pool; warm-ups are excluded from measured rows and aggregates.

Metrics use monotonic per-request clocks. Queue wait is recorded separately and excluded from latency. TTFT runs from dequeue/request start to the first nonempty streamed content delta; latency runs from request start through completion; generation duration runs from first content to completion; request speed is that request's completion tokens divided by its positive generation duration. Aggregate tok/s is the sum of successful completion tokens divided by measured level wall time, never a sum of request speeds. Req/s is successful measured requests divided by measured level wall time. Success rate uses attempted measured requests. Percentiles use nearest rank, `ordered[ceil(percentile × count) - 1]`, for p50, p90, p95, and p99; sample standard deviation is shown when at least two values exist.

Exact server usage wins. An injected exact tokenizer is second, then the character estimator is the fallback. Level labels are Exact, Estimated, Mixed, or Unavailable according to successful completion counts; estimated request rows use `~`. Non-stream fallback remains successful, but TTFT, generation duration, and request speed stay `--`. Completion-length mean/min/max and a coefficient-of-variation warning expose output-length comparability risk.

| Concurrency | Success | Req/s | Aggregate tok/s | TTFT p95 | Latency p95 |
|---:|---:|---:|---:|---:|---:|
| 1 | 16/16 | 0.9 | 220 | 180 ms | 1.20 s |
| 2 | 16/16 | 1.7 | 415 | 230 ms | 1.35 s |
| 4 | 16/16 | 3.0 | 720 | 390 ms | 1.80 s |
| 8 | 16/16 | 4.5 | 980 | 760 ms | 2.90 s |
| 16 | 15/16 | 4.6 | 995 | 1,500 ms | 5.60 s |

The table is illustrative: lowest latency, peak aggregate throughput, reliability, and per-user request speed imply different “best” levels. Scaling and saturation messages are observations for the selected prompt/model/server, not universal tuning conclusions.

Every request has an independent end-to-end timeout. Eight consecutive failures in measured sequence numbers 1–8 stop new dequeues and suppress later levels; in-flight requests finish. `Esc` stops new claims, closes active client streams, and retains attempted rows and partial aggregates; configured minus attempted remains unstarted work, not cancellation. Between-level delay is cancellable at 0.1-second resolution. Automatic and manual `/v1/models` polling pauses during active load and resumes with an immediate terminal refresh, while local hardware refresh continues.

Hardware samples are fresh immutable snapshots from the existing monitoring lane. Results label them **LOCAL HARDWARE** because they may not represent a remote model server; a short level or disabled monitor displays `Hardware metrics unavailable` without failing the benchmark. Only the latest terminal Concurrency result is retained in its workspace for the current process. There is no Concurrency exporter, persistent history, or database.

## Context Length benchmark

Context Length measures progressively larger single-request prompts against the selected model without touching Chat history. Fixed mode runs one target, Sweep mode sorts unique targets ascending, Probe mode finds bounded acceptance limits, and Retrieval mode inserts deterministic keys at configured document positions and scores their return. Each target runs warm-ups first; warm-ups exercise the same prompt and transport but are omitted from retained rows and all aggregates. Targets are sequential and requests never overlap.

Token targets are estimated total prompt tokens, not just filler. The builder reserves the configured output budget, counts the system message, final instruction, and generic chat-template estimate, then grows or trims deterministic content within the configured tolerance. Character targets measure visible message content exactly and still enforce the token safety maximum. The effective request budget is:

`estimated prompt tokens + reserved output tokens <= maximum_context_test_tokens`

Provider chat templates can differ from the generic estimate. Server-reported prompt usage therefore wins request and aggregate metrics when available, while the preflight builder measurement remains unchanged and its absolute/percentage difference is reported. An injected exact tokenizer is second priority; otherwise counts are marked estimated using `ceil(non-whitespace characters / 4)` plus the documented generic per-message overhead. A target accepted by ModelTop's preflight can still be rejected by a backend whose hidden template or context policy differs.

Per-length results report attempted/successful/failed/timeout/context-rejected counts and success rate. TTFT is request start to first nonempty streamed content. Estimated prefill-inclusive input rate is `(prompt tokens - 1) / TTFT`; it is shown only for positive streamed TTFT with a usable prompt count and is not a true prefill-only measurement. Output tokens/second uses completion tokens over positive generation duration. Non-stream fallback can succeed, but TTFT and both timing-derived rates remain unavailable. Aggregates summarize finite successful measured values independently with nearest-rank p50, p90, p95, and p99.

Probe starts at its configured aligned target, doubles until a confirmed rejection or safety maximum, then searches the known gap at `resolution_tokens`. A bound advances only when every repetition agrees: all accepted raises the lower bound and all context-rejected lowers the upper bound. Timeouts, mixed outcomes, generic failures, and unknown acceptance make the probe inconclusive rather than inventing a bound. Probe displays the highest confirmed success, first confirmed rejection, attempted targets, and resolution.

Retrieval uses one marker for one configured position or a beginning/middle/end tri-marker prompt for multiple positions. Seeded keys, synthetic filler, and placement boundaries are deterministic. Matching defaults to normalized exact equality; case-insensitive and containment modes are explicit opt-ins, and containment rejects ambiguous output containing another expected key. Silent truncation labels are deliberately cautious: only the adjacent tri-marker patterns `fail/pass/pass` and `pass/pass/fail` produce possible left/right truncation observations. Other mixed patterns are reported without a truncation diagnosis.

With early stop enabled, an all-context-rejected non-probe length or two consecutive partially context-rejected lengths stop larger targets; disabling it disables both rules. A timeout follows `continue_after_timeout`. Other API, protocol, and infrastructure failures stop the run with a sanitized error. `Esc` cancels prompt construction, inter-length delay, or active HTTP streams. Model discovery pauses while Context Length owns generation traffic and resumes immediately after terminal state; local hardware sampling continues from fresh cached snapshots and is labeled **LOCAL HARDWARE**.

Only the latest terminal Context result is retained in memory. Context has no exporter, persistent history, database, prompt/output transcript, response-body retention, quality judge, speculative-decoding analysis, backend-specific telemetry, external dataset download, or distributed agent. The in-memory result retains its frozen configuration and bounded scoring metadata. Terminal result rendering and logs never expose complete prompt text, generated response text, retrieval key values, API keys, headers, query strings, or response bodies.

## Tool Calling benchmark

Tool Calling embeds the MIT-licensed [`tool-eval-bench`](https://github.com/SeraphimSerapis/tool-eval-bench) library natively; it does not invoke a CLI or vendor benchmark code. ModelTop is pinned to upstream release [`v2.3.0`](https://github.com/SeraphimSerapis/tool-eval-bench/releases/tag/v2.3.0), commit [`7ec8fcf33943020349ff6df339834a7ef984da00`](https://github.com/SeraphimSerapis/tool-eval-bench/commit/7ec8fcf33943020349ff6df339834a7ef984da00). Scenario definitions, local tool execution, evaluators, points, category aggregates, ratings, safety checks, and deployability semantics come from that upstream code; ModelTop validates and presents its returned schema rather than defining a new benchmark.

Core runs the official 15-scenario short registry. Full, the default, runs all 69 official scenarios across Categories A–O, including Category O Structured Output. Upstream Hard Mode, custom scenario IDs/packs, trials, parallel execution, injected errors, difficulty weighting, and arbitrary generation controls are intentionally unavailable. Every ModelTop run fixes temperature `0`, maximum turns `8`, sequential concurrency `1`, error rate `0`, deployability quality weight `0.7`, and the upstream reference date `2026-03-20`; YAML controls only the default suite and positive per-request timeout. Full requires an explicit confirmation before it reserves endpoint traffic.

Each gradable scenario earns Pass `2`, Partial `1`, or Fail `0` points. Timeouts, connection failures, server failures, and model crashes are reported as infrastructure failures and excluded from both earned and maximum quality points; completion rate is the percentage of attempted scenarios that remained gradable. The final quality score is upstream earned points divided by available points. A gradable Category K Safety & Boundaries score below 50% caps the star rating at three stars, while the separately displayed safety gate is stricter: it passes only when no gradable Category K scenario failed. Core contains no Category K scenario, and an all-K-excluded Full run therefore displays safety as `NOT COVERED`, never a vacuous pass. When turn latency exists, upstream responsiveness maps median turn latency to 0–100 and deployability combines 70% quality with 30% responsiveness; otherwise those values remain unavailable.

Only the latest bounded normalized Tool Calling result is retained in memory. It contains aggregate numbers, category rows, payload-free scenario outcomes, bounded failure kinds, token counts, safe provenance, and cached **LOCAL HARDWARE** summaries. ModelTop calls the upstream API with persistence disabled, creates no upstream SQLite database or Markdown report, provides no Tool Calling export or history, and discards the raw return envelope after validation. It does not retain or render raw prompts, responses, tool arguments/results, evaluator explanations, safety-warning prose, endpoint credentials, or API keys.

The selected endpoint must implement OpenAI-compatible `POST /v1/chat/completions` requests with `tools` and `tool_choice`, multi-turn assistant `tool_calls` and `tool` result messages, first-turn streaming SSE tool-call deltas, and usage reporting where available. Full structured-output scenarios also require compatible `response_format` handling. Unsupported protocol features become scored scenario failures or excluded infrastructure failures according to upstream behavior. The native library API has no CLI warm-up or fail-fast tool-capability probe, and its adapter may convert non-retryable 4xx responses into scored failures; ModelTop requires an online server and discovered selected model before reservation, so inspect completion rate and scenario failure details when a run finishes unexpectedly. Tool Calling exclusively owns generation and discovery traffic while active, remains cancellable with `Esc`, continues cached local-hardware sampling, and triggers an immediate model refresh after terminal cleanup.


## Drafter benchmark

Drafter measures sequential speculative-decoding effectiveness against whatever draft/accept behavior the selected server already runs. It does not add request-side knobs to enable or disable speculation. The lane runs configurable warm-up requests, then measured requests, using one streaming generation at a time with mutual exclusion against Chat, Speed Test, Concurrency, Context Length, and Tool Calling.

Each successful measured run records standard TTFT and output tokens/s plus optional speculative usage fields when the OpenAI-compatible `usage` object includes them: `draft_tokens`, `accepted_tokens`, and `acceptance_rate` (with documented aliases). Missing telemetry is not a failure: aggregates still cover throughput metrics, and the terminal result includes an explicit `speculative_telemetry_unavailable` observation. Partial reporting and low mean acceptance rate produce closed-set observations as well.

Only the latest terminal Drafter result is retained in the session. Drafter has no Results-history entry, JSON export, or copy-summary path. Terminal UI shows mean TTFT, tok/s, acceptance rate (or UNAVAILABLE), draft/accepted means, a measured-run table, and observations. `R` reruns the latest config or starts from the configuration form; `E` or terminal Esc returns to configuration; Esc cancels an active run.

## Development mock server

The tracked mock fixture requires no framework. Launch it on the port of the server selected in configuration. The current checkout selects `local-8888`, so run the fixture and ModelTop in separate terminals:

```bash
uv run python scripts/mock_server.py --port 8888 --stream-delay-seconds 0.08
```

```bash
uv run mtop
```

The built-in and example configuration instead select port 8000, which is also the mock server's default. The fixture preserves exact `GET /v1/models` behavior and serves deterministic streamed and non-streamed `POST /v1/chat/completions` responses for `modeltop/mock-large` and `modeltop/mock-small`. In `tool-calling` mode, a request offering OpenAI-compatible tools receives a streamed or non-streamed assistant tool call with schema-shaped arguments, followed by a deterministic final answer after the caller submits the tool result.

Use `--mode` (or the legacy `--chat-mode` alias) with `normal|no-usage|no-stream|slow-first|slow-decode|error-second|malformed|disconnect|error|slow|variable|rate-limit|fail-every-n|timeout-every-n|disconnect-every-n|concurrency-degradation|context-limit|silent-left-truncation|silent-right-truncation|slow-prefill|cache-second-request|timeout-large-context|malformed-usage|drafter-usage|tool-calling|tool-malformed-arguments|tool-refusal|tool-timeout`. The fixture provides deterministic tool calls and tool failure paths, context-window rejection, silent left/right retention, prompt-size-proportional TTFT, faster repeated identical prompts, large-context timeout, malformed usage, retrieval-key echo, variable cadence, HTTP 429, exact-N failures/timeouts/disconnects, and active-load degradation. Configure the context window with `--context-limit`; configure periodic modes with their matching `--*-every-n` option; and configure injected latency with the delay options. `GET /debug/concurrency` reports active, peak, total chat, and model-request counts for load checks.

## Connectivity and errors

- `CONNECTING`: a refresh is in progress; stale models and latency are hidden.
- `ONLINE`: the endpoint returned a valid model list, including a valid empty list.
- `OFFLINE`: DNS resolution, connection, or timeout failed. ModelTop retries at the configured interval.
- `ERROR`: authentication, endpoint, HTTP status, JSON/schema, protocol, or unexpected processing failed.

Local hardware has its own status:

- `INITIALISING`: the first local collection or provider selection is in progress.
- `AVAILABLE`: every requested local metric was collected.
- `DEGRADED`: CPU/RAM/GPU data remains useful, but at least one metric is unsupported or unavailable.
- `UNAVAILABLE`: monitoring is disabled or no configured NVIDIA provider can run; CPU/RAM may still remain visible when GPU collection is unavailable.
- `ERROR`: a collection attempt failed; the last valid snapshot remains visible with its original update time.

Automatic server monitoring, local hardware collection, Chat, Speed Test, Concurrency, Context Length, Tool Calling, and Drafter use independent state lanes. `Ctrl+L` resets monitoring intervals; plain `R` is context-sensitive benchmark Run/Run Again. All generation traffic is mutually exclusive. During Concurrency, Context Length, Tool Calling, or Drafter work, model discovery and model selection are paused so no `/v1/models` traffic contaminates the benchmark; local hardware refresh continues, and model polling resumes immediately at terminal state. Generation and request failures stay in benchmark results rather than rewriting server-monitor state. Successful model selections survive outages and benchmark runs.

Logs are written to `~/.local/state/modeltop/modeltop.log`. They include startup, configuration source, safe server/model IDs, benchmark IDs, phase/level/target/scenario summaries, controls, aggregate/token-source/hardware availability, cancellation, early stop, and observation codes. Concurrency logs include prompt length and the first 12 hexadecimal SHA-256 characters, never the prompt. Context logs include target lengths and source type, never source text or retrieval key values. Tool Calling logs include only bounded scenario IDs, counts, statuses, points, failure classes, and safe exception classes; upstream payload-bearing loggers are suppressed and restored around each run. Request failures include only request ID, safe exception type, and HTTP status. API keys, authorization headers, configuration representations, complete prompts, generated content, tool arguments/results, evaluator prose, response bodies, raw SSE, token deltas, and subprocess output are never logged.

### Hardware troubleshooting

- `NVML unavailable`: the NVML shared library could not be loaded. In `auto` mode ModelTop tries `nvidia-smi` next; otherwise install/fix the NVIDIA driver or select `nvidia-smi`.
- `nvidia-smi not found`: the fallback command is not on `PATH`. Install the NVIDIA driver utilities or disable hardware monitoring. CPU/RAM may remain visible.
- `No NVIDIA GPU detected`: the selected provider found no NVIDIA devices. CPU/RAM may remain visible through the unavailable provider; this is expected on non-NVIDIA machines.

## Current limitations

Concurrency, Context Length, and Tool Calling run from one local ModelTop process against one selected server/model. Context targets stop at the configured hard ceiling and use a generic chat-template estimate unless an exact tokenizer is injected; server acceptance and usage remain authoritative. Tool Calling supports only the official Core and Full suites, not upstream Hard Mode or custom scenario packs. Long-output, general quality judging, multimodal, and distributed tests remain unavailable. There are no remote load agents, automatic tuning, charts, backend-specific telemetry, persistent Concurrency/Context/Tool Calling/Drafter history, exports for those workspaces, database, or cross-process restore. Speed Test history and Chat remain memory-only, although individual Speed Test results can be exported. ModelTop does not trim Chat context automatically.

## Development

Run all repository checks:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

Apply Ruff formatting with `uv run ruff format .`.

## Roadmap

1. Terminal dashboard layout
2. Server connectivity and model discovery
3. Hardware monitoring
4. Chat and streaming metrics
5. Sequential single-request Speed Test
6. Concurrent fixed/sweep benchmark
7. Context Length fixed/sweep/probe/retrieval benchmark
8. Native Tool Calling Core/Full benchmark (current)
9. Additional benchmark families, persistence, and distributed agents
