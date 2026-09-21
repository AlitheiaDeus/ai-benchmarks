#!/usr/bin/env python3
"""
LM Studio Automated Benchmark Suite v2.0
========================================
Tag-Delimited Multi-Model Benchmark Engine for LM Studio.

Key Features:
- Tag-delimited input suite parsing (<suite>, <task id="..." category="...">, <prompt>...</prompt>)
- Tag-delimited CDATA-isolated output reports (.tags) & companion clean Markdown reports (.md)
- Multi-model batch selection & sequential execution in a single run
- Prompt suite source file selector (default: UNCENSOR_ME.txt)
- Blank system prompt injection ({"role": "system", "content": ""}) to neutralize LM Studio defaults
- Thinking / reasoning mode toggles (enabled / disabled with proper token headroom management)
- Clickable Textual TUI (benchmark_ui.py) + robust interactive CLI fallback
"""

import os
import sys
import json
import time
import re
import argparse
from datetime import datetime
from urllib import request, error
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Ensure UTF-8 output handling on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


DEFAULT_BASE_URL = "http://localhost:1234"
DEFAULT_PROMPT_FILE = "suite_math_logic.txt"
DEFAULT_OUTPUT_DIR = "benchmarks"


def query_api(url: str, payload: dict = None, timeout: Optional[float] = None) -> dict:
    """Send an HTTP JSON request using standard library urllib (timeout=None means no timeout)."""
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = request.Request(url, data=data, headers=headers, method="POST" if payload else "GET")
    
    try:
        kwargs = {"timeout": timeout} if timeout is not None else {}
        with request.urlopen(req, **kwargs) as response:
            res_body = response.read().decode("utf-8")
            return json.loads(res_body)
    except error.HTTPError as e:
        err_msg = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} Error: {err_msg}")
    except error.URLError as e:
        raise RuntimeError(f"Could not connect to LM Studio at {url}. Reason: {e.reason}")


def get_available_models(base_url: str = DEFAULT_BASE_URL) -> List[str]:
    """Fetch the list of downloaded / loaded models from LM Studio."""
    discovered = []
    
    # 1. Attempt standard OpenAI-compatible /v1/models endpoint (5s probe timeout)
    models_url = f"{base_url.rstrip('/')}/v1/models"
    try:
        data = query_api(models_url, timeout=5.0)
        if "data" in data and isinstance(data["data"], list):
            discovered = [m.get("id") for m in data["data"] if m.get("id")]
    except Exception:
        pass

    # 2. Fallback to LM Studio native /api/v1/models endpoint (5s probe timeout)
    if not discovered:
        native_url = f"{base_url.rstrip('/')}/api/v1/models"
        try:
            data = query_api(native_url, timeout=5.0)
            if "models" in data and isinstance(data["models"], list):
                discovered = [m.get("id") or m.get("name") for m in data["models"] if (m.get("id") or m.get("name"))]
        except Exception:
            pass

    # Exclude nomic embedding model
    return [m for m in discovered if "text-embedding-nomic-embed-text-v1.5" not in m.lower()]


def find_available_prompt_suites(search_dir: str = ".") -> List[Path]:
    """Scans directory for available benchmark prompt suite files (.txt, .tags, .xml)."""
    suite_files = []
    p = Path(search_dir)
    for ext in ("*.txt", "*.tags", "*.xml"):
        for f in p.glob(ext):
            if f.is_file() and not f.name.startswith("benchmark_results_") and not f.name.startswith("benchmark_batch_"):
                suite_files.append(f)
    
    # Sort with DEFAULT_PROMPT_FILE first if it exists
    suite_files.sort(key=lambda x: (x.name != DEFAULT_PROMPT_FILE, x.name.lower()))
    return suite_files


def parse_prompts_file(filepath: str) -> List[Dict[str, Any]]:
    """
    Parses prompts from a pure tag-delimited file structure:
    <suite>
      <task id="1" category="...">
        <prompt>...</prompt>
      </task>
    </suite>
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Prompt suite file not found: {filepath}")

    raw_text = path.read_text(encoding="utf-8")
    
    # Regex parser for robust tag extraction
    task_pattern = re.compile(
        r'<task(?:\s+[^>]*)?>.*?</task>',
        re.DOTALL | re.IGNORECASE
    )
    
    tasks = []
    for idx, match in enumerate(task_pattern.finditer(raw_text), 1):
        task_block = match.group(0)
        
        # Extract ID
        id_match = re.search(r'\bid\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^\s>]+))', task_block, re.IGNORECASE)
        task_id = (id_match.group(1) or id_match.group(2) or id_match.group(3)) if id_match else str(idx)
        
        # Extract Category
        cat_match = re.search(r'\bcategory\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^\s>]+))', task_block, re.IGNORECASE)
        category = (cat_match.group(1) or cat_match.group(2) or cat_match.group(3)) if cat_match else f"Task {idx}"
        
        # Extract Criteria
        crit_match = re.search(r'<criteria>(.*?)</criteria>', task_block, re.DOTALL | re.IGNORECASE)
        criteria = crit_match.group(1).strip() if crit_match else ""
        
        # Extract Prompt
        prompt_match = re.search(r'<prompt>(.*?)</prompt>', task_block, re.DOTALL | re.IGNORECASE)
        prompt_text = prompt_match.group(1) if prompt_match else ""
        
        # Clean CDATA markers if present inside prompt
        cdata_match = re.search(r'<!\[CDATA\[(.*?)\]\]>', prompt_text, re.DOTALL)
        if cdata_match:
            prompt_text = cdata_match.group(1)

        tasks.append({
            "index": idx,
            "id": task_id.strip(),
            "category": category.strip(),
            "criteria": criteria.strip(),
            "prompt": prompt_text.strip()
        })

    if not tasks:
        raise ValueError(
            f"No valid <task><prompt>...</prompt></task> blocks found in '{filepath}'. "
            "Please ensure the file uses the tag-delimited structure."
        )

    return tasks


def extract_reasoning_and_output(response_json: dict) -> Tuple[str, str]:
    """
    Extracts reasoning/thinking text and the final response across different model families:
    - Gemma-4 / Gemma-2: <|channel>thought...<channel|>, <start_of_thought>...<end_of_thought>
    - Qwen-3.5 / DeepSeek: <think>...</think>
    - Llama / Custom: <thought>, [THINK], reasoning_content field
    """
    choices = response_json.get("choices", [])
    if not choices:
        return "", response_json.get("error", {}).get("message", "No response choices returned.")

    message = choices[0].get("message", {})
    raw_content = message.get("content", "") or ""
    
    # 1. Direct reasoning fields
    api_reasoning = message.get("reasoning_content") or message.get("thought") or message.get("reasoning") or ""
    
    thinking_segments = []
    if api_reasoning.strip():
        thinking_segments.append(api_reasoning.strip())

    # 2. Tag-based reasoning delimiters
    thinking_patterns = [
        r"<\|channel\>thought\s*(.*?)(?:<channel\|>|<\|channel\>|$)",
        r"<\|channel\|>thought\s*(.*?)(?:<\|channel\|>|$)",
        r"<start_of_thought>\s*(.*?)(?:<end_of_thought>|$)",
        r"<\|thought_start\|>\s*(.*?)(?:<\|thought_end\|>|$)",
        r"<\|thought\|>\s*(.*?)(?:<\|\/thought\|>|<\|thought\|>|$)",
        r"<\|start_header_id\|>thought<\|end_header_id\|>\s*(.*?)(?:<\|eot_id\|>|<\|start_header_id\|>|$)",
        r"<think>\s*(.*?)(?:</think>|$)",
        r"<thought>\s*(.*?)(?:</thought>|$)",
        r"<reasoning>\s*(.*?)(?:</reasoning>|$)",
        r"\[THINK\]\s*(.*?)(?:\[/THINK\]|$)",
        r"\[THOUGHT\]\s*(.*?)(?:\[/THOUGHT\]|$)",
        r"\[REASONING\]\s*(.*?)(?:\[/REASONING\]|$)",
        r"```(?:thought|thinking|reasoning)\s*\n(.*?)(?:```|$)",
    ]

    clean_content = raw_content
    for pattern in thinking_patterns:
        matches = re.findall(pattern, clean_content, flags=re.DOTALL | re.IGNORECASE)
        for match in matches:
            if match.strip():
                thinking_segments.append(match.strip())
        clean_content = re.sub(pattern, "", clean_content, flags=re.DOTALL | re.IGNORECASE)

    clean_content = clean_content.strip()
    full_thinking = "\n\n---\n\n".join(thinking_segments).strip()
    
    return full_thinking, clean_content


def execute_single_task(
    base_url: str,
    model_id: str,
    task_item: Dict[str, Any],
    enable_thinking: bool = True,
    system_prompt_mode: str = "blank",
    custom_system_prompt: str = "",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    ttl: Optional[int] = None
) -> Dict[str, Any]:
    """Executes a single benchmark task against LM Studio API with no request timeout."""
    chat_url = f"{base_url.rstrip('/')}/v1/chat/completions"
    p_text = task_item["prompt"]
    
    messages = []
    
    # 1. System Prompt Injection
    if system_prompt_mode == "blank":
        messages.append({"role": "system", "content": ""})
    elif system_prompt_mode == "custom" and custom_system_prompt:
        messages.append({"role": "system", "content": custom_system_prompt})
    # If "none", omit system message entirely to preserve model-configured LM Studio defaults

    messages.append({"role": "user", "content": p_text})

    # 2. Clean OpenAI API Payload (preserves LM Studio presets if not explicitly overridden)
    payload = {
        "model": model_id,
        "messages": messages,
        "stream": False,
    }

    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if ttl is not None:
        payload["ttl"] = ttl
    if enable_thinking is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": bool(enable_thinking)}

    req_start = time.time()
    error_msg = None
    thinking_text = ""
    response_text = ""
    usage = {}
    finish_reason = "stop"
    status = "SUCCESS"

    try:
        # timeout=None waits indefinitely for the model to complete without artificial cutoff
        res_json = query_api(chat_url, payload=payload, timeout=None)
        req_duration = time.time() - req_start
        
        thinking_text, response_text = extract_reasoning_and_output(res_json)
        usage = res_json.get("usage", {})
        
        choices = res_json.get("choices", [])
        if choices:
            finish_reason = choices[0].get("finish_reason", "stop")
            
    except Exception as e:
        req_duration = time.time() - req_start
        status = "FAILED"
        error_msg = str(e)
        response_text = f"**ERROR:** {error_msg}"

    return {
        "index": task_item["index"],
        "id": task_item.get("id", str(task_item["index"])),
        "category": task_item["category"],
        "criteria": task_item.get("criteria", ""),
        "prompt": p_text,
        "duration": req_duration,
        "status": status,
        "finish_reason": finish_reason,
        "thinking": thinking_text,
        "response": response_text,
        "usage": usage,
        "error": error_msg
    }





def generate_markdown_report(
    model_id: str,
    prompt_file: str,
    total_time: float,
    results: List[Dict[str, Any]],
    enable_thinking: bool,
    system_prompt_mode: str
) -> str:
    """Formats benchmark results into a clean, structured Markdown report."""
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    lines = [
        f"# Benchmark Report: {model_id}",
        "",
        f"- **Date & Time:** {date_str}",
        f"- **Model ID:** `{model_id}`",
        f"- **Prompt Suite:** `{prompt_file}`",
        f"- **Total Duration:** {total_time:.2f} seconds",
        f"- **Thinking Mode:** {'Enabled' if enable_thinking else 'Disabled'}",
        f"- **System Prompt Mode:** `{system_prompt_mode}`",
        "",
        "## Summary Scorecard",
        "",
        "| Task # | Category | Status | Latency (s) | Thinking (chars) | Response (chars) | Total Tokens | Finish Reason |",
        "|:---|:---|:---|:---|:---|:---|:---|:---|",
    ]

    for r in results:
        status_icon = "âœ… Success" if r["status"] == "SUCCESS" else "âŒ Failed"
        total_tokens = r["usage"].get("total_tokens", "-") if r["usage"] else "-"
        lines.append(
            f"| {r['index']} | {r['category']} | {status_icon} | {r['duration']:.2f} | "
            f"{len(r['thinking']):,} | {len(r['response']):,} | {total_tokens} | `{r.get('finish_reason', 'stop')}` |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Detailed Task Responses")
    lines.append("")

    for r in results:
        lines.append(f"### Task {r['index']}: {r['category']}")
        lines.append("")
        lines.append(f"**Execution Time:** {r['duration']:.2f}s | **Status:** `{r['status']}` | **Finish Reason:** `{r.get('finish_reason', 'stop')}`")
        lines.append("")
        
        # User Prompt (Collapsible)
        lines.append("<details>")
        lines.append(f"<summary><b>View Prompt ({len(r['prompt'])} chars)</b></summary>")
        lines.append("")
        lines.append("```text")
        lines.append(r["prompt"])
        lines.append("```")
        lines.append("</details>")
        lines.append("")

        # Thinking Process (Collapsible)
        if r["thinking"]:
            lines.append("<details open>")
            lines.append(f"<summary><b>ðŸ’­ Model Thinking Process ({len(r['thinking']):,} chars)</b></summary>")
            lines.append("")
            lines.append("```text")
            lines.append(r["thinking"])
            lines.append("```")
            lines.append("")
            lines.append("</details>")
            lines.append("")
        else:
            lines.append("> *[No explicit thinking process or reasoning tokens captured for this task]*")
            lines.append("")

        # Model Response
        lines.append("#### ðŸ’¬ Model Final Response:")
        lines.append("")
        lines.append(r["response"])
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def save_model_reports(
    model_id: str,
    prompt_file: str,
    total_time: float,
    results: List[Dict[str, Any]],
    enable_thinking: bool = True,
    system_prompt_mode: str = "blank",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    output_base: str = None
) -> List[str]:
    """Generates and writes a single structured .md report for a model into output_dir."""
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    clean_model_name = re.sub(r'[^\w\-_\.]', '_', model_id)
    clean_suite_name = re.sub(r'[^\w\-_\.]', '_', Path(prompt_file).stem)
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_filename = output_base or f"{clean_model_name}_{clean_suite_name}_{timestamp_str}"
    
    md_report = generate_markdown_report(
        model_id=model_id,
        prompt_file=prompt_file,
        total_time=total_time,
        results=results,
        enable_thinking=enable_thinking,
        system_prompt_mode=system_prompt_mode
    )

    md_file = out_dir / f"{base_filename}.md"
    md_file.write_text(md_report, encoding="utf-8")

    return [str(md_file)]


def select_models_interactive(models: List[str]) -> List[str]:
    """CLI interactive multi-model selection prompt."""
    if not models:
        print("\n[!] No models were automatically discovered from LM Studio.")
        manual_id = input("Enter Model ID / Path manually (comma-separated for multiple): ").strip()
        if not manual_id:
            print("Error: No model ID provided.")
            sys.exit(1)
        return [m.strip() for m in manual_id.split(",") if m.strip()]

    print("\n" + "=" * 65)
    print("  AVAILABLE MODELS IN LM STUDIO (Batch Selection)")
    print("=" * 65)
    for idx, model_name in enumerate(models, 1):
        print(f"  [{idx}] {model_name}")
    print("  [A] All models")
    print("=" * 65)

    while True:
        choice = input("\nSelect models [e.g. '1', '1,3,4', 'all']: ").strip()
        if not choice:
            continue
        if choice.lower() in ("a", "all"):
            return models
        
        selected = []
        parts = [p.strip() for p in choice.split(",") if p.strip()]
        valid = True
        for part in parts:
            if part.isdigit() and 1 <= int(part) <= len(models):
                selected.append(models[int(part) - 1])
            else:
                # Support case-insensitive partial / substring matching
                matched = [m for m in models if part.lower() in m.lower()]
                if matched:
                    selected.extend(matched)
                elif len(part) > 1:
                    selected.append(part)
                else:
                    valid = False
                    break
        
        if valid and selected:
            # Deduplicate while preserving order
            return list(dict.fromkeys(selected))
        print("Invalid selection. Please enter valid numbers or model names.")


def select_prompt_file_interactive(default_file: str = DEFAULT_PROMPT_FILE) -> str:
    """CLI interactive prompt suite file selection."""
    files = find_available_prompt_suites()
    if not files:
        return default_file

    print("\n" + "=" * 65)
    print("  AVAILABLE PROMPT SUITES")
    print("=" * 65)
    for idx, f in enumerate(files, 1):
        is_def = " (default)" if f.name == default_file else ""
        print(f"  [{idx}] {f.name}{is_def}")
    print("=" * 65)

    choice = input(f"\nSelect suite [1-{len(files)}] (Press Enter for '{default_file}'): ").strip()
    if not choice:
        return default_file
    if choice.isdigit() and 1 <= int(choice) <= len(files):
        return str(files[int(choice) - 1])
    return choice


def run_batch_benchmark(
    base_url: str,
    model_ids: List[str],
    prompt_file: str,
    enable_thinking: bool = True,
    system_prompt_mode: str = "blank",
    custom_system_prompt: str = "",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    ttl: Optional[int] = None,
    output_dir: str = DEFAULT_OUTPUT_DIR
):
    """Executes sequential batch benchmarking across all selected models."""
    tasks = parse_prompts_file(prompt_file)
    if not tasks:
        print(f"[!] No valid tasks found in '{prompt_file}'.")
        return

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    temp_disp = f"{temperature}" if temperature is not None else "MODEL_DEFAULT"
    tokens_disp = f"{max_tokens}" if max_tokens is not None else "MODEL_DEFAULT"
    ttl_disp = f"{ttl}s" if ttl is not None else "SERVER_DEFAULT"

    print("\n" + "=" * 65)
    print("  LM STUDIO BATCH BENCHMARK RUNNER")
    print("=" * 65)
    print(f"[*] Total Models       : {len(model_ids)} ({', '.join(model_ids)})")
    print(f"[*] Prompt Suite       : {prompt_file} ({len(tasks)} tasks)")
    print(f"[*] Output Directory   : {out_path.resolve()}")
    print(f"[*] Thinking Mode      : {'ENABLED' if enable_thinking else 'DISABLED'}")
    print(f"[*] System Prompt Mode : {system_prompt_mode.upper()}")
    print(f"[*] Temp / Max Tokens  : {temp_disp} | {tokens_disp}")
    print(f"[*] JIT Model TTL      : {ttl_disp}")
    print(f"[*] Endpoint           : {base_url}/v1/chat/completions")
    print("=" * 65)

    batch_start_time = time.time()
    batch_results = {}

    for m_idx, model_id in enumerate(model_ids, 1):
        print(f"\n\n{'#' * 65}")
        print(f"  [{m_idx}/{len(model_ids)}] BENCHMARKING MODEL: {model_id}")
        print(f"{'#' * 65}")

        model_start_time = time.time()
        model_task_results = []

        for task in tasks:
            print(f"\n[>] Task {task['index']}/{len(tasks)}: {task['category']} ...")
            
            task_res = execute_single_task(
                base_url=base_url,
                model_id=model_id,
                task_item=task,
                enable_thinking=enable_thinking,
                system_prompt_mode=system_prompt_mode,
                custom_system_prompt=custom_system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                ttl=ttl
            )
            model_task_results.append(task_res)

            status_icon = "âœ“" if task_res["status"] == "SUCCESS" else "âœ—"
            print(
                f"    {status_icon} Completed in {task_res['duration']:.2f}s "
                f"| Thinking: {len(task_res['thinking']):,} chars "
                f"| Response: {len(task_res['response']):,} chars "
                f"| Finish: {task_res.get('finish_reason', 'stop')}"
            )

        model_duration = time.time() - model_start_time
        batch_results[model_id] = {
            "duration": model_duration,
            "results": model_task_results
        }

        # Save individual model reports
        saved = save_model_reports(
            model_id=model_id,
            prompt_file=prompt_file,
            total_time=model_duration,
            results=model_task_results,
            enable_thinking=enable_thinking,
            system_prompt_mode=system_prompt_mode,
            temperature=temperature,
            max_tokens=max_tokens,
            output_dir=output_dir
        )
        print(f"\n[âœ“] Saved Model Report: {', '.join(saved)}")

    total_batch_time = time.time() - batch_start_time

    print("\n" + "=" * 65)
    print("  ALL BENCHMARKS COMPLETED SUCCESSFULLY")
    print("=" * 65)
    print(f"  Total Duration : {total_batch_time:.2f}s")
    print(f"  Models Run     : {len(model_ids)}")
    print(f"  Total Tasks    : {len(model_ids) * len(tasks)}")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="LM Studio Automated Benchmark Suite v2.0 with Batch Execution & Tag-Delimited Reports."
    )
    parser.add_argument(
        "--url", "-u",
        default=DEFAULT_BASE_URL,
        help=f"LM Studio base URL (default: {DEFAULT_BASE_URL})"
    )
    parser.add_argument(
        "--models", "-m",
        default=None,
        help="Comma-separated model IDs or paths (if omitted, interactive selector is shown)"
    )
    parser.add_argument(
        "--file", "-f",
        default=DEFAULT_PROMPT_FILE,
        help=f"Path to prompt suite file (default: {DEFAULT_PROMPT_FILE})"
    )
    parser.add_argument(
        "--select-file",
        action="store_true",
        help="Interactively select prompt suite file from available files"
    )
    parser.add_argument(
        "--thinking",
        dest="thinking",
        action="store_true",
        default=True,
        help="Enable model thinking / reasoning mode (default: enabled)"
    )
    parser.add_argument(
        "--no-thinking",
        dest="thinking",
        action="store_false",
        help="Disable model thinking / reasoning mode"
    )
    parser.add_argument(
        "--system-prompt-mode", "-s",
        choices=["blank", "none", "custom"],
        default="blank",
        help="System prompt injection mode: 'blank' (neutralize with blank system prompt, default), 'none' (preserve LMS presets), or 'custom'"
    )
    parser.add_argument(
        "--system-prompt",
        default="",
        help="Custom system prompt text (used when --system-prompt-mode custom)"
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=None,
        help="Generation temperature (default: None, preserves LM Studio / model presets)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum tokens to generate (default: None, preserves LM Studio / model context limits)"
    )
    parser.add_argument(
        "--ttl",
        type=int,
        default=None,
        help="LM Studio JIT model idle lifespan / Time-To-Live in seconds (e.g. --ttl 5 or --ttl 300)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to store generated benchmark reports (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--list-models", "-l",
        action="store_true",
        help="List available models in LM Studio and exit"
    )
    args = parser.parse_args()

    # Model Listing Option
    if args.list_models:
        available_models = get_available_models(args.url)
        if not available_models:
            print(f"\n[!] No models found or could not connect to LM Studio at {args.url}")
        else:
            print("\n" + "=" * 65)
            print("  AVAILABLE MODELS IN LM STUDIO")
            print("=" * 65)
            for idx, m in enumerate(available_models, 1):
                print(f"  [{idx}] {m}")
            print("=" * 65 + "\n")
        return

    # 1. Source File Selection
    if args.select_file:
        prompt_file = select_prompt_file_interactive(args.file)
    else:
        prompt_file = args.file

    # 2. Model Selection
    available_models = get_available_models(args.url)

    if args.models:
        if args.models.lower() in ("a", "all"):
            selected_models = available_models if available_models else [args.models]
        else:
            raw_parts = [m.strip() for m in args.models.split(",") if m.strip()]
            selected_models = []
            for part in raw_parts:
                if available_models:
                    if part.isdigit() and 1 <= int(part) <= len(available_models):
                        selected_models.append(available_models[int(part) - 1])
                        continue
                    # Case-insensitive substring matching
                    matched = [m for m in available_models if part.lower() in m.lower()]
                    if matched:
                        selected_models.extend(matched)
                        continue
                selected_models.append(part)
            selected_models = list(dict.fromkeys(selected_models))
    else:
        selected_models = select_models_interactive(available_models)

    if not selected_models:
        print("[!] No models selected to benchmark.")
        return

    # 3. Run Benchmark
    run_batch_benchmark(
        base_url=args.url,
        model_ids=selected_models,
        prompt_file=prompt_file,
        enable_thinking=args.thinking,
        system_prompt_mode=args.system_prompt_mode,
        custom_system_prompt=args.system_prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        ttl=args.ttl,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
