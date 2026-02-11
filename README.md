# Coding Agent

A small LLM-powered coding agent that can run tools (execute code, read/write files) via the OpenAI-compatible API (e.g. OpenRouter). It supports both **models with native tool calling** (e.g. Nemotron) and **models without** (e.g. Gemma), plus an optional **planning** mode that first generates a step-by-step plan, then executes it.

---

## Project layout

| File / folder      | Description |
|--------------------|-------------|
| `agent.py`         | LLM client wrapper, simple coding agent, and planning agent. |
| `tools.py`         | Tool implementations and OpenAI-style schemas. | 
| `.env`              | `OPENROUTER_API_KEY` (Visit https://openrouter.ai/ to create free API key). |

---

## Tools (`tools.py`)

Tools are plain Python functions plus an OpenAI-style schema each (name, description, parameters). The agent passes these to the LLM and executes the chosen tool with the LLM-provided arguments.

### Available tools

| Tool            | Description | Main parameters |
|-----------------|-------------|------------------|
| **`execute_code`** | Run Python code in-process and return stdout/errors. | `code` (string). |
| **`read_file`**   | Read file contents with optional `limit` and `offset`. | `file_path` (required), `limit`, `offset`. |
| **`write_file`**  | Write content to a path; creates parent directories if needed. Handles `\n`/`\t` escape sequences for multi-line content. | `content`, `file_path`. |

### Schemas

Each tool has a `*_schema` dict (e.g. `execute_code_schema`, `read_file_schema`, `write_file_schema`) in the format expected by the Chat Completions API:

- `type`: `"function"`
- `function`: `name`, `description`, `parameters` (JSON Schema with `properties`, `required`).

These are used in `agent.py` both for **native** tool calling (passed as `tools` to the API) and for **prompt-injected** tool descriptions (see below).

### Errors

- `ToolError` is raised for invalid paths or permission issues; the agent wraps tool results and reports errors back to the LLM.

---

## Agent (`agent.py`)

The agent uses a single **LLM wrapper** and two high-level modes: **simple coding agent** and **planning agent**. Behavior depends on whether the model supports **native** tool use or not.

### Models: native vs non-native tools

| Kind | Examples | How tools are used |
|------|----------|---------------------|
| **Native tool support** | e.g. `nvidia/nemotron-3-nano-30b-a3b:free` | API `tools` and `tool_choice` are set; the model returns structured `tool_calls`; the agent executes them and passes results back in the conversation. |
| **No native tools** | e.g. `google/gemma-3-12b-it:free`, `deepseek/deepseek-r1` | No `tools` in the request. Tool names and parameters are **injected into the system/user prompt**. The model is asked to output a tool call in a fixed text format; the agent **parses** that (e.g. `<tool_call>...</tool_call>` or ` ```tool_call ... ``` `) and runs the tool, then continues the loop. |

The list of “no native tools” models is in `MODELS_WITHOUT_NATIVE_TOOLS` (e.g. `"google/gemma-3-"`, `"deepseek/deepseek-r1"`). For these models the code also avoids sending a `system` role where the provider disallows it (e.g. Gemma “developer instruction”); the system text is injected as a user message instead.

---

### Simple coding agent: `coding_agent(...)`

- **Purpose:** One query → multiple rounds of tool use until the model responds without a tool call or `max_tool_rounds` is reached.
- **Flow:**
  1. Build messages with a system prompt (and, for non-native models, inject tool descriptions).
  2. Call the LLM once.
  3. **Native:** If the response has `message.tool_calls`, execute each, append tool results to messages, call the LLM again; repeat.
  4. **Prompt-injected:** Parse the response for a single `<tool_call>` (or ` ```tool_call ``` `) block; if found, execute that tool, append the result as a user message, call the LLM again; repeat.
  5. When there are no tool calls (or parse fails), the last assistant message is treated as the final answer.
- **Parameters:** `client`, `query`, `system`, `tools` (dict of callables), `tools_schemas` (list of schema dicts), `name` (model id), `max_tool_rounds`.

Colored state prints (e.g. `[LLM-START]`, `[LLM-NATIVE]`, `[LLM-PROMPT]`, `[TOOL]`) indicate which path is used and when tools run.

---

### Planning agent: `coding_agent_with_planning(...)`

- **Purpose:** First **plan** (sequence of tools and reasons), then **execute** each step by asking the LLM for arguments and running the tool. Good for multi-file or multi-step tasks.
- **Flow:**
  1. **Plan phase:** `create_plan(client, query, tools_schemas, model)` asks the LLM for a JSON list of steps (each step: `tool`, `reason`; no `args` — the LLM fills those at execution time).
  2. **Execute phase:** `execute_plan_iteratively(...)` for each step: send “use tool X for this step”, get LLM response (native `tool_calls` or prompt-injected `<tool_call>`), execute the tool, add result to the conversation, then next step.
  3. After all steps, the LLM is asked for a short summary of what was done.
- **Parameters:** Same as `coding_agent` plus `use_planning=True` and `working_dir` (passed into the executor’s system prompt). If planning is disabled or the plan is empty, the code can fall back to standard (non-planning) behavior.

Colored state prints use labels like `[PLAN]`, `[EXEC]`, `[EXEC-TOOL]`, `[PLANNING]` so you can see plan creation, each step, tool calls, and the final summary.

---

### Helper pieces in `agent.py`

- **`llm(client, messages, system, name, tools, **kwargs)`**  
  Single place for Chat Completions calls: handles system vs user-injected system text for Gemma-like models, and only adds `tools`/`tool_choice` when the model has native tool support.

- **`tools_schemas_to_prompt_text(tools_schemas)`**  
  Turns the list of tool schemas into a text block for the system (or user) prompt when using prompt-injected tools.

- **`parse_tool_call_from_text(text)`**  
  Extracts one tool call from model output. Accepts both `<tool_call>{"name":..., "arguments":...}</tool_call>` and ` ```tool_call ... ``` `.

- **`execute_tool(name, args, tools)`**  
  Dispatches to the right callable in `tools`; `args` can be a JSON string or a dict.

---

## Quick start

1. Set `OPENROUTER_API_KEY` in `.env` (or configure your client’s `base_url` and API key).
2. From the `coding_agent` directory run:

   ```bash
   python agent.py
   ```

   The `if __name__ == "__main__"` block uses a default model and query (e.g. Snake game in `agent_files/`). You can switch the model (e.g. Nemotron vs Gemma) and toggle `coding_agent` vs `coding_agent_with_planning(..., use_planning=True)` there.

3. For **native** tool models, use a model id that supports function calling (e.g. Nemotron). For **Gemma** (or others in `MODELS_WITHOUT_NATIVE_TOOLS`), the agent will use prompt-injected tools and the same entrypoint.

---

## Summary

| Component | Role |
|-----------|------|
| **tools.py** | Defines `execute_code`, `read_file`, `write_file` and their OpenAI-style schemas. |
| **agent.py** | LLM wrapper + simple coding agent + planning agent; supports native and prompt-injected tools; uses colored prints for state. |
| **Native tools** | Models that support API `tools` (e.g. Nemotron); agent uses `message.tool_calls`. |
| **Non-native tools** | Models like Gemma; agent injects tool docs into the prompt and parses `<tool_call>` (or ```tool_call```) from the reply. |
| **Simple agent** | `coding_agent`: query → repeated tool rounds → final answer. |
| **Planning agent** | `coding_agent_with_planning`: create plan (tool sequence) → execute each step with LLM-chosen arguments → final summary. |

*[reference sources](https://www.deeplearning.ai/short-courses/building-coding-agents-with-tool-execution/)*: 