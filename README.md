# LM Studio Automated Benchmark Suite (`ai-benchmarks`)

A high-precision, tag-delimited multi-model benchmarking engine for local LLMs running on [LM Studio](https://lmstudio.ai/).

---

## Key Features

- **Tag-Delimited Suite Architecture**: Uses XML-style tags (`<suite>`, `<task id="..." category="...">`, `<criteria>`, `<prompt>`) to define reproducible, structured evaluation tasks.
- **CDATA-Isolated Reports**: Produces isolated `.tags` execution traces alongside clean Markdown (`.md`) summary reports.
- **Thinking / Headroom Management**: Native support for toggling internal `<think>` / reasoning traces while automatically managing context window and token headrooms.
- **System Neutralization**: Injects blank system roles to neutralize default UI/system prompt biases for pure raw model completions.
- **Sequential Multi-Model Testing**: Run multiple loaded models back-to-back against the same evaluation suite.

---

## Included Benchmark Suites

1. **`suite_math_logic.txt` — Advanced Math & Formal Logic**
   - **Number Theory**: Tetration modular arithmetic (Euler's Totient Theorem, Chinese Remainder Theorem).
   - **Combinatorics**: Binomial derangements and probability reductions.
   - **Formal Logic**: Truth deductions and multi-step reasoning.

2. **`suite_prose_depth.txt` — Prose & Stylistic Depth**
   - **Authorial Voice**: Cormac McCarthy polysyndeton, rhythmic sensory weight, anti-cliché filters.
   - **Negative Constraints**: Hardboiled noir detective scenes with strict forbidden word lists (no delve, tapestry, neon, shadows).

3. **`suite_python_scripting.txt` — Python Automation & Scripting**
   - **Log Parsing & Regex**: Web access log parsing, metric aggregation, and structured JSON output.
   - **Filesystem & Hashing**: Recursive deduplication with chunked SHA-256 and space recovery reporting.
   - **CLI Data Wrangling**: Standalone CLI flattening arbitrary nested JSON to filtered, projected CSVs.

4. **`suite_enterprise_wrangling.txt` — Enterprise Legacy Data Wrangling (JavaScript / Node.js)**
   - **SAP Spool Parsing**: Fixed-width positional slicing, wrapped continuation row stitching, SAP trailing minus parsing (`1,250.00-`), and pagination banner rejection.
   - **Spreadsheet Unpivoter**: Normalizing multi-tier grouped headers (`Region -> Metric -> Actual/Target`) and forward-filling merged empty cells into flat database records.
   - **EDI / X12 850 Parser**: Dynamic delimiter discovery and state-machine loop parsing of ANSI X12 purchase orders into nested JSON.

---

## Quick Start

### 1. Requirements
- Python 3.10+ (Recommended Conda env: `textrpg`)
- LM Studio running local server (`http://localhost:1234`)

```bash
conda activate textrpg
# Or in a fresh environment:
pip install requests textual
```

### 2. Run Benchmark
Run the default suite (`suite_math_logic.txt`):
```bash
python benchmark.py
```

Specify a custom suite or parameters:
```bash
python benchmark.py --prompts suite_prose_depth.txt --temperature 0.7
```

### 3. Turn Extraction Utility
Extract individual model turn answers cleanly from execution runs:
```bash
python extract_model_turns.py
```
