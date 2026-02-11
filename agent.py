from dotenv import load_dotenv
import os
import re
from typing import TypedDict, Optional
import sys
from io import StringIO
import json
from typing import Callable
from openai import OpenAI
_ = load_dotenv()
from tools import execute_code, execute_code_schema, read_file, read_file_schema, write_file, write_file_schema

# Models that don't support native tool/function calling (use prompt-injected tools).
MODELS_WITHOUT_NATIVE_TOOLS = ("google/gemma-3-", "deepseek/deepseek-r1")

# Simple ANSI colors for debug output
COLOR_RESET = "\033[0m"
COLOR_CYAN = "\033[36m"
COLOR_YELLOW = "\033[33m"
COLOR_MAGENTA = "\033[35m"
COLOR_GREEN = "\033[32m"
COLOR_RED = "\033[31m"


def _print_state(label: str, detail: str = "", color: str = COLOR_CYAN) -> None:
    """Colored debug print to show LLM / agent state."""
    prefix = f"[{label}]"
    msg = f"{prefix} {detail}" if detail else prefix
    print(f"{color}{msg}{COLOR_RESET}")


def tools_schemas_to_prompt_text(tools_schemas: list[dict]) -> str:
    """Turn OpenAI-style tool schemas into a text block for the system prompt."""
    lines = [
        "You have access to the following tools. When you need to use one, output a single block in this exact format (one line of JSON, no extra text inside the block):",
        "",
        "<tool_call>",
        '{"name": "<tool_name>", "arguments": { ... }}',
        "</tool_call>",
        "",
        "Available tools:",
    ]
    for schema in tools_schemas:
        fn = schema.get("function") or {}
        name = fn.get("name", "?")
        desc = fn.get("description", "")
        params = fn.get("parameters") or {}
        props = params.get("properties") or {}
        required = params.get("required") or []
        lines.append(f"- {name}: {desc}")
        for k, v in props.items():
            req = " (required)" if k in required else ""
            lines.append(f"  - {k} ({v.get('type', 'any')}): {v.get('description', '')}{req}")
        lines.append("")
    lines.append(
        "When calling a tool, output only one <tool_call>...</tool_call> block with valid JSON. "
        + "After the tool runs, you will receive the result and can respond or call another tool."
    )
    return "\n".join(lines)


def parse_tool_call_from_text(text: str) -> Optional[tuple[str, dict]]:
    """Extract a single tool call from model output. Returns (tool_name, arguments) or None."""
    if not text:
        return None
    # Prefer the explicit <tool_call> ... </tool_call> format
    match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
    # Some models instead use a fenced code block ```tool_call ... ```
    if not match:
        match = re.search(r"```tool_call\s*(\{.*?\})\s*```", text, re.DOTALL)
        if not match:
            return None
    try:
        payload = json.loads(match.group(1).strip())
        name = payload.get("name")
        args = payload.get("arguments")
        if isinstance(args, dict) and name:
            return (name, args)
        return None
    except (json.JSONDecodeError, TypeError):
        return None


def llm(
    client,
    messages: list[dict],
    system: str = "You are an helpful assistant",
    name: str = "google/gemma-3-12b-it:free",
    tools: list[dict] = [],
    **kwargs,
):
    # Build message list with or without a true system message,
    # depending on what the target model supports.
    final_messages = list(messages)
    if system:
        # Gemma via Google AI Studio does not support developer/system
        # instructions for some models (e.g. gemma-3-12b-it). For those,
        # inject the instructions as an extra user message instead.
        if name.startswith("google/gemma-3-"):
            final_messages = [
                {"role": "user", "content": system},
                *final_messages,
            ]
        else:
            final_messages = [
                {"role": "system", "content": system},
                *final_messages,
            ]

    # Models like Gemma don't support native tool calling; don't pass tools.
    use_native_tools = tools and not any(
        name.startswith(prefix) for prefix in MODELS_WITHOUT_NATIVE_TOOLS
    )
    request_kwargs = dict(
        model=name,
        messages=final_messages,
        **kwargs,
    )
    if use_native_tools:
        request_kwargs["tools"] = tools
        request_kwargs["tool_choice"] = "auto"
    response = client.chat.completions.create(**request_kwargs)
    return response


def execute_tool(name: str, args: str | dict, tools: dict[str, Callable]):
    """Execute a tool with given arguments.
    
    Args:
        name: Tool name
        args: Either a JSON string or a dict of arguments
        tools: Dictionary of available tool functions
    """
    try:
        # If args is already a dict, use it directly; otherwise parse JSON
        if isinstance(args, str):
            args = json.loads(args)
        
        if name not in tools:
            return {"error": f"Tool {name} doesn't exist."}
        result = tools[name](**args)
    except json.JSONDecodeError as e:
        result = {"error": f"{name} failed to parse arguments: {str(e)}"}
    except KeyError as e:
        result = {"error": f"Missing key in arguments: {str(e)}"}
    except Exception as e:
        result = {"error": str(e)}
    return result


def _use_prompt_injected_tools(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in MODELS_WITHOUT_NATIVE_TOOLS)


def coding_agent(
    client: OpenAI,
    query: str,
    system: str,
    tools: dict[str, Callable],
    tools_schemas: list[dict],
    name: str = "google/gemma-3-12b-it:free",
    max_tool_rounds: int = 5,
):
    use_prompt_tools = _use_prompt_injected_tools(name)
    if use_prompt_tools:
        system_with_tools = system.rstrip() + "\n\n" + tools_schemas_to_prompt_text(tools_schemas)
    else:
        system_with_tools = system

    _print_state(
        "LLM-START",
        f"model={name}, prompt_tools={use_prompt_tools}",
        COLOR_CYAN,
    )

    messages = [{"role": "user", "content": query}]
    response = llm(client, messages, system_with_tools, name=name, tools=tools_schemas if not use_prompt_tools else [])
    choice = response.choices[0]
    message = choice.message
    content = message.content or ""

    # For native tools, include tool_calls; for prompt tools, don't
    if not use_prompt_tools:
        messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": getattr(message, "tool_calls", None),
        })
    else:
        messages.append({
            "role": "assistant",
            "content": content,
        })

    # Native tool calls (models that support function calling) — loop until no more tool_calls or max_tool_rounds
    if not use_prompt_tools:
        rounds = 0
        while rounds < max_tool_rounds:
            _print_state(
                "LLM-NATIVE",
                f"round={rounds}, tool_calls={len(message.tool_calls or []) if getattr(message, 'tool_calls', None) is not None else 0}",
                COLOR_YELLOW,
            )
            if not message.tool_calls:
                _print_state("LLM-NATIVE", "no tool_calls, final answer", COLOR_GREEN)
                print(content)
                return
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                arguments = tool_call.function.arguments
                _print_state("TOOL", f"{tool_name} executing", COLOR_MAGENTA)
                result = execute_tool(tool_name, arguments, tools)
                print(f"[{tool_name}] {result}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                })
            _print_state("LLM-NATIVE", "calling LLM with tool results", COLOR_CYAN)
            response = llm(client, messages, system_with_tools, name=name, tools=tools_schemas)
            message = response.choices[0].message
            content = message.content or ""
            messages.append({
                "role": "assistant",
                "content": content,
                "tool_calls": getattr(message, "tool_calls", None),
            })
            rounds += 1
        _print_state("LLM-NATIVE", "max_tool_rounds reached, printing last content", COLOR_RED)
        print(content)
        return

    # Prompt-injected tools: parse <tool_call> from assistant text
    if use_prompt_tools:
        rounds = 0
        while rounds < max_tool_rounds:
            _print_state(
                "LLM-PROMPT",
                f"round={rounds}, attempting to parse <tool_call>",
                COLOR_YELLOW,
            )
            parsed = parse_tool_call_from_text(content)
            if not parsed:
                _print_state("LLM-PROMPT", "no <tool_call> found, final answer", COLOR_GREEN)
                print(content)
                return
            tool_name, arguments = parsed
            _print_state("TOOL", f"{tool_name} executing", COLOR_MAGENTA)
            # Pass arguments dict directly
            result = execute_tool(tool_name, arguments, tools)
            print(f"[{tool_name}] {result}")
            messages.append({
                "role": "user",
                "content": f"Tool result for {tool_name}: {json.dumps(result)}. Continue: if the task is done, reply with a short summary and no <tool_call> block; otherwise output another <tool_call> block.",
            })
            _print_state("LLM-PROMPT", "calling LLM with tool result", COLOR_CYAN)
            response = llm(client, messages, system_with_tools, name=name, tools=[])
            message = response.choices[0].message
            content = message.content or ""
            messages.append({"role": "assistant", "content": content})
            rounds += 1
        _print_state("LLM-PROMPT", "max_tool_rounds reached, printing last content", COLOR_RED)
        print("Final response:", content)
        return

    _print_state("LLM", "no tools used, direct answer", COLOR_GREEN)
    print(content)


def create_plan(client, query: str, tools_schemas: list[dict], model: str) -> list[dict]:
    """Create a simple execution plan for the query.
    
    Returns list of steps: [{"step": 1, "tool": "tool_name", "args": {...}, "reason": "..."}, ...]
    """
    # Build tool descriptions
    tools_desc = []
    for schema in tools_schemas:
        fn = schema.get("function", {})
        name = fn.get("name", "?")
        desc = fn.get("description", "")
        params = fn.get("parameters", {}).get("properties", {})
        required = fn.get("parameters", {}).get("required", [])
        
        tool_info = f"- {name}: {desc}\n"
        for param_name, param_info in params.items():
            req = " (required)" if param_name in required else ""
            tool_info += f"  * {param_name}: {param_info.get('description', '')}{req}\n"
        tools_desc.append(tool_info)
    
    planning_prompt = f"""Create a step-by-step plan to accomplish this task: {query}

Available tools:
{chr(10).join(tools_desc)}

Output ONLY valid JSON (no markdown, no explanation):
{{
  "steps": [
    {{"step": 1, "tool": "tool_name", "reason": "why this step"}},
    {{"step": 2, "tool": "tool_name", "reason": "why this step"}}
  ]
}}

Rules:
- DO NOT include "args" in the plan - just tool name and reason
- The agent will figure out arguments during execution
- Keep it simple: just the sequence of tools to use"""

    _print_state("PLAN", f"creating plan for query (model={model})", COLOR_CYAN)
    try:
        _print_state("PLAN", "calling LLM for plan JSON", COLOR_YELLOW)
        response = llm(
            client,
            messages=[{"role": "user", "content": planning_prompt}],
            system="You are a planning assistant. Output only valid JSON with tool sequence.",
            name=model,
            tools=[],
            temperature=0.3,
        )
        content = response.choices[0].message.content
        
        # Extract JSON from response
        try:
            plan_data = json.loads(content.strip())
        except json.JSONDecodeError:
            json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
            if json_match:
                plan_data = json.loads(json_match.group(1).strip())
            else:
                json_match = re.search(r'\{.*"steps".*\}', content, re.DOTALL)
                if json_match:
                    plan_data = json.loads(json_match.group(0))
                else:
                    _print_state("PLAN", "Failed to extract JSON from response", COLOR_RED)
                    return []

        steps = plan_data.get("steps", [])
        _print_state("PLAN", f"Generated {len(steps)} steps", COLOR_GREEN)
        for step in steps:
            _print_state("PLAN", f"  Step {step.get('step')}: {step.get('tool')} - {step.get('reason', '')}", COLOR_GREEN)
        return steps

    except Exception as e:
        _print_state("PLAN", f"Error: {e}", COLOR_RED)
        return []


def execute_plan_iteratively(
    client: OpenAI,
    plan: list[dict],
    query: str,
    system: str,
    tools: dict[str, Callable],
    tools_schemas: list[dict],
    model: str,
    working_dir: str = "",
) -> str:
    """Execute plan step-by-step, asking LLM for tool arguments at each step.
    
    This is the TRUE planning approach:
    1. Create high-level plan (tool sequence)
    2. For each step, ask LLM what arguments to use
    3. Execute tool with those arguments
    4. Continue to next step
    """
    import json
    
    messages = []
    results_summary = []
    
    # Add system message
    system_msg = f"""{system}

You are executing a plan step-by-step. For each step:
1. I'll tell you which tool to use and why
2. You call that tool with appropriate arguments
3. I'll show you the result
4. We move to the next step

Original task: {query}
Working directory: {working_dir}"""
    
    use_prompt_tools = _use_prompt_injected_tools(model)
    if use_prompt_tools:
        system_with_tools = system_msg.rstrip() + "\n\n" + tools_schemas_to_prompt_text(tools_schemas)
    else:
        system_with_tools = system_msg
    
    # Initial user message
    messages.append({
        "role": "user",
        "content": f"We have a {len(plan)}-step plan to complete this task: {query}\n\nLet's start with step 1."
    })
    
    _print_state("EXEC", f"starting iterative execution of {len(plan)} steps (model={model})", COLOR_CYAN)

    # Execute each step
    for i, step in enumerate(plan, 1):
        tool_name = step.get("tool", "")
        reason = step.get("reason", "")

        _print_state("EXEC", f"Step {i}/{len(plan)}: {reason}", COLOR_YELLOW)
        _print_state("EXEC", f"Tool to use: {tool_name}", COLOR_MAGENTA)

        # Ask LLM to execute this specific step
        step_instruction = f"\nStep {i}: {reason}\nPlease use the '{tool_name}' tool now."
        messages.append({"role": "user", "content": step_instruction})

        _print_state("EXEC", f"calling LLM for step {i}", COLOR_CYAN)
        # Get LLM response with tool call
        response = llm(
            client,
            messages,
            system_with_tools,
            name=model,
            tools=tools_schemas if not use_prompt_tools else []
        )
        
        message = response.choices[0].message
        content = message.content or ""
        
        # Handle tool calls (native or prompt-injected)
        if not use_prompt_tools and message.tool_calls:
            # Native tool calling
            tool_call = message.tool_calls[0]  # Take first tool call
            actual_tool = tool_call.function.name
            arguments = tool_call.function.arguments

            _print_state("EXEC-TOOL", f"LLM calling {actual_tool}", COLOR_MAGENTA)
            _print_state("EXEC-TOOL", f"Args: {arguments}", COLOR_CYAN)
            
            # Add assistant message
            messages.append({
                "role": "assistant",
                "content": content,
                "tool_calls": [{
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": actual_tool,
                        "arguments": arguments
                    }
                }]
            })
            
            # Execute tool
            result = execute_tool(actual_tool, arguments, tools)
            result_str = str(result)
            preview = result_str[:300] if len(result_str) > 300 else result_str
            _print_state("EXEC-TOOL", f"Result: {preview}{'...' if len(result_str) > 300 else ''}", COLOR_GREEN)

            # Add tool result
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })

            results_summary.append(f"Step {i} ({actual_tool}): Success")

        elif use_prompt_tools:
            # Prompt-injected tool calling
            messages.append({"role": "assistant", "content": content})

            parsed = parse_tool_call_from_text(content)
            if parsed:
                actual_tool, arguments = parsed
                _print_state("EXEC-TOOL", f"LLM calling {actual_tool}", COLOR_MAGENTA)
                _print_state("EXEC-TOOL", f"Args: {arguments}", COLOR_CYAN)

                # Execute tool
                result = execute_tool(actual_tool, arguments, tools)
                result_str = str(result)
                preview = result_str[:300] if len(result_str) > 300 else result_str
                _print_state("EXEC-TOOL", f"Result: {preview}{'...' if len(result_str) > 300 else ''}", COLOR_GREEN)
                
                # Add result as user message
                messages.append({
                    "role": "user",
                    "content": f"Tool result: {json.dumps(result)}\n\n" + (
                        f"Good! Now let's continue to step {i+1}." if i < len(plan) 
                        else "Great! All steps completed."
                    )
                })
                
                results_summary.append(f"Step {i} ({actual_tool}): Success")
            else:
                _print_state("EXEC", "No tool call found in response", COLOR_RED)
                results_summary.append(f"Step {i}: No tool call")
        else:
            _print_state("EXEC", "No tool calls in response", COLOR_RED)
            messages.append({"role": "assistant", "content": content})
            results_summary.append(f"Step {i}: No tool call")

    # Final summary
    _print_state("EXEC", "Execution summary", COLOR_CYAN)
    for summary in results_summary:
        _print_state("EXEC", f"  ✓ {summary}", COLOR_GREEN)

    # Ask LLM for final summary
    _print_state("EXEC", "Asking LLM for final summary", COLOR_YELLOW)
    messages.append({
        "role": "user",
        "content": f"All {len(plan)} steps completed. Please provide a brief summary of what was accomplished."
    })
    
    try:
        final_response = llm(
            client,
            messages,
            system_with_tools,
            name=model,
            tools=[]
        )
        _print_state("EXEC", "Final summary received", COLOR_GREEN)
        return final_response.choices[0].message.content
    except Exception as e:
        _print_state("EXEC", f"Summary LLM failed ({e}), using fallback", COLOR_RED)
        return f"Completed {len(plan)} steps successfully."

def coding_agent_with_planning(
    client: OpenAI,
    query: str,
    system: str,
    tools: dict[str, Callable],
    tools_schemas: list[dict],
    name: str = "google/gemma-3-12b-it:free",
    max_tool_rounds: int = 5,
    use_planning: bool = False,
    working_dir: str = "",
):
    """
    Coding agent with optional planning mode.
    
    Args:
        use_planning: If True, creates a plan first then executes it
        working_dir: Working directory for file operations
    """
    if use_planning:
        _print_state("PLANNING", "PLANNING MODE", COLOR_CYAN)
        _print_state("PLANNING", f"query: {query[:80]}{'...' if len(query) > 80 else ''}", COLOR_YELLOW)

        # PHASE 1: Create plan (quick, just tool sequence)
        plan = create_plan(client, query, tools_schemas, name)

        if not plan:
            _print_state("PLANNING", "No plan created, falling back to standard mode", COLOR_RED)
            use_planning = False
        else:
            # PHASE 2: Execute plan iteratively (LLM decides args at each step)
            _print_state("PLANNING", f"EXECUTING {len(plan)}-STEP PLAN", COLOR_MAGENTA)
            result = execute_plan_iteratively(
                client, plan, query, system, tools, tools_schemas, name, working_dir
            )
            _print_state("PLANNING", "FINAL SUMMARY", COLOR_GREEN)
            print(result)
            return result

    # Standard mode (fall through to your original coding_agent)
    _print_state("PLANNING", "Standard mode (no planning)", COLOR_YELLOW)
    use_prompt_tools = _use_prompt_injected_tools(name)
    if use_prompt_tools:
        system_with_tools = system.rstrip() + "\n\n" + tools_schemas_to_prompt_text(tools_schemas)
    else:
        system_with_tools = system
    # Standard mode (original coding_agent logic)

if __name__ == "__main__":
    print("="*50, "Testing the agent...", "="*50)
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    # Try a faster model that supports tools better
    NEMOTRON_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"
    DEEPSEEK_MODEL = "deepseek/deepseek-r1-0528:free"
    GEMMA_MODEL = "google/gemma-3-12b-it:free"

    model_name = NEMOTRON_MODEL
    print("="*50, "Testing write file...", "="*50)
    working_dir = os.path.abspath(os.getcwd() + "/agent_files")
    os.makedirs(working_dir, exist_ok=True)

    tools = {
        "execute_code": execute_code,
        "read_file": read_file,
        "write_file": write_file,
    }
    
#     # ✅ Simpler query first to test
#     query = f"""Create a blank text.txt file in {working_dir}"""
    
#     # ✅ FIX: Update system prompt to mention ALL available tools
#     system = f"""You are a helpful coding assistant.

# Available tools and when to use them:
# - execute_code: Run Python code to perform calculations or operations
# - write_file: Create or update files (REQUIRED for creating any file)
# - read_file: Read contents of existing files

# IMPORTANT: 
# - To create ANY file, you MUST use the write_file tool
# - For write_file, always use absolute paths starting with: {working_dir}
# - Example: To create text.txt, use path: {working_dir}/text.txt

# Work step by step and use the appropriate tools."""
    query = f"""Create a web project in {working_dir}:

1. Create a minimalist Snake game and save it into snake.html
It must have the following features:
- 30x30 grid (CSS Grid or Canvas)
- Arrow keys for movement
- Food spawns randomly
- Snake grows when eating and food disappears
- Game over on wall/self collision
- Simple score counter
- Restart button
- Retro green-on-black styling
- Pure vanilla JS, no libs
- Snake speed is 4 block per second
- Use random food emoji for food

2. Create a README.md explaining how to play

3. Create a Python script launcher.py that opens the game in a browser
"""
    
    system = f"""You are a helpful coding assistant specializing in web development. Do the user's task step by step."""

    # coding_agent(
    #     client,
    #     query,
    #     system,
    #     tools=tools,
    #     tools_schemas=[execute_code_schema,
    #                 read_file_schema,
    #                 write_file_schema],
    #     name=model_name,
    #     max_tool_rounds=10,
    # )
    # # Use planning mode
    coding_agent_with_planning(
        client,
        query,
        system,
        tools=tools,
        tools_schemas=[execute_code_schema, read_file_schema, write_file_schema],
        name=model_name,
        use_planning=True,  # Enable planning
        working_dir=working_dir,
    )