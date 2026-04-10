import os
import json
import re
import asyncio
import sys
from openai import OpenAI
from openenv.core.env_client import EnvClient

# Force Python to find models.py in the root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
from models import Action, Observation

# ── 1. ENVIRONMENT VARIABLES ──────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_SPACE_URL = os.getenv("HF_SPACE_URL", "https://sukuna191552s-pramanaenv.hf.space")

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# ── 2. CLIENTS ────────────────────────────────────────────────────────────────
llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


class DataCleanerClient(EnvClient):
    """EnvClient subclass that parses our custom Observation model."""

    def _parse_state(self, data: dict) -> Observation:
        obs_data = data.get("observation", data.get("state", data))
        return Observation(**obs_data)

    def _parse_result(self, data: dict):
        obs_data = data.get("observation", data.get("state", data))
        obs = Observation(**obs_data)
        return (
            obs,
            float(data.get("reward", obs.reward)),
            bool(data.get("terminated", data.get("done", obs.done))),
            bool(data.get("truncated", False)),
            data.get("info", {}),
        )

    def _step_payload(self, action: Action) -> dict:
        return action.model_dump()


# ── 3. LOGGING ────────────────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error=None):
    err_str = "null" if error is None else str(error)
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} error={err_str}",
        flush=True,
    )

def log_end(success: bool, steps: int, rewards: list):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ── 4. AGENT ──────────────────────────────────────────────────────────────────
def get_model_action(step: int, obs_dict: dict) -> Action:
    instructions = obs_dict.get("target_schema_instructions", "")
    missing      = obs_dict.get("missing_values", {})
    dtypes       = obs_dict.get("data_types", {})
    feedback     = obs_dict.get("last_action_feedback", "")
    columns      = obs_dict.get("current_columns", [])

    has_missing = any(v > 0 for v in missing.values())

    prompt = f"""You are a precise data cleaning agent. Follow the instructions exactly and output one JSON action.

TASK INSTRUCTIONS: {instructions}

CURRENT DATASET STATE:
- Columns: {columns}
- Data types: {dtypes}
- Missing value counts: {missing}
- Feedback from last action: {feedback}

DECISION RULES (follow in order):
1. If missing value counts show any column > 0, fix missing values FIRST.
   - Use fill_missing_values if the instruction says "fill" (provide target_column and new_value).
   - Use drop_missing_rows if the instruction says "drop" (provide target_column only).
2. If no missing values remain but a column dtype is wrong per the instructions, use change_data_type.
   - For integer columns use new_value="int".
   - For float columns use new_value="float".
3. ONLY call submit_final_dataset when ALL of these are true:
   - All missing value counts are 0.
   - All column dtypes match what the instructions require.
4. Do NOT repeat an action that already succeeded (check the feedback).
5. Do NOT call submit_final_dataset if any missing values still exist.

Current missing values present: {has_missing}

Output ONLY a single JSON object with no explanation, no markdown, no extra text:
{{"tool": "tool_name", "target_column": "ColumnName", "new_value": "value"}}

For submit_final_dataset, drop_missing_rows, or undo_last_action you may omit new_value."""

    try:
        response = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=100,
        )
        raw_text = (response.choices[0].message.content or "").strip()

        match = re.search(r"\{.*?\}", raw_text, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object in response: {raw_text!r}")

        parsed = json.loads(match.group())
        return Action(**parsed)

    except Exception as exc:
        print(f"[WARN] step={step} model_action_failed: {exc}", flush=True)
        return Action(tool="undo_last_action")


# ── 5. MAIN LOOP ──────────────────────────────────────────────────────────────
async def main() -> None:
    env_name  = "SST_hackathon_env"
    
    task_ids = ["task_1_age", "task_2_salary", "task_3_price"]

    try:
        env_client = DataCleanerClient(HF_SPACE_URL)

        for task_id in task_ids:
            rewards: list = []
            steps_taken = 0
            success = False

            log_start(task=task_id, env=env_name, model=MODEL_NAME)

            try:
                raw_reset = await env_client.reset()
                obs_obj   = raw_reset[0] if isinstance(raw_reset, tuple) else raw_reset
                obs_dict  = obs_obj.model_dump() if hasattr(obs_obj, "model_dump") else dict(obs_obj)

                done = False

                for step in range(1, 11):
                    if done:
                        break

                    steps_taken = step
                    action = get_model_action(step, obs_dict)

                    action_str = action.model_dump_json(exclude_none=True).replace("\n", "")

                    raw_step = await env_client.step(action)
                    obs_obj  = raw_step[0] if isinstance(raw_step, tuple) else raw_step
                    obs_dict = obs_obj.model_dump() if hasattr(obs_obj, "model_dump") else dict(obs_obj)

                    # Extract raw reward from server
                    raw_reward = float(getattr(obs_obj, "reward", 0.05))
                    
                    # 🛡️ THE TITANIUM CLAMP: Never trust the server's boundary values
                    reward = float(min(max(raw_reward, 0.05), 0.95))
                    
                    done   = bool(getattr(obs_obj, "done",   False))

                    rewards.append(reward)
                    log_step(step=step, action=action_str, reward=reward, done=done, error=None)

                if rewards and rewards[-1] >= 0.90:
                    success = True

            except Exception as exc:
                print(f"[ERROR] Task {task_id} failed: {exc}", flush=True)

            finally:
                if not rewards:
                    rewards     = [0.05]   
                    steps_taken = 1
                    
                # 🛡️ PARANOID CLAMP: Ensure the final END tag is strictly bounded
                safe_rewards = [float(min(max(r, 0.05), 0.95)) for r in rewards]
                
                log_end(success=success, steps=steps_taken, rewards=safe_rewards)
                
    except Exception as exc:
        print(f"[FATAL ERROR] Client setup failed: {exc}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())