import gymnasium as gym
import pandas as pd
import numpy as np
from pydantic import BaseModel

# --- PYDANTIC MODELS ---
class Action(BaseModel):
    tool: str
    target_column: str = ""
    new_value: str = ""

class Observation(BaseModel):
    columns: list
    null_counts: dict
    dtypes: dict
    target_schema_instructions: str
    last_action_feedback: str

# --- THE REAL ENVIRONMENT ---
class DataCleanerEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # DEFINING 3 DISTINCT TASKS FOR THE SCALER BOT
        self.tasks = [
            {
                "name": "task_1_age",
                "data": pd.DataFrame({"Age": [25, np.nan, 30], "Name": ["Alice", "Bob", "Charlie"]}),
                "instructions": "Fill missing Age with 25."
            },
            {
                "name": "task_2_salary",
                "data": pd.DataFrame({"Salary": [50000, 60000, np.nan], "Department": ["IT", "HR", "IT"]}),
                "instructions": "Drop rows with missing Salary."
            },
            {
                "name": "task_3_price",
                "data": pd.DataFrame({"Price": ["10", "20", "30"], "Item": ["Apple", "Banana", "Cherry"]}),
                "instructions": "Change Price data type to int."
            }
        ]
        self.current_task_idx = 0
        self.dataframe = None
        self.instructions = ""
        self.step_count = 0

    def reset(self, seed=None, options=None):
        self.step_count = 0
        task = self.tasks[self.current_task_idx]
        self.dataframe = task["data"].copy()
        self.instructions = task["instructions"]
        
        # Cycle to the next task for the next reset
        self.current_task_idx = (self.current_task_idx + 1) % len(self.tasks)
        
        return self._get_obs("Environment initialized."), {}

    def _get_obs(self, feedback: str) -> Observation:
        return Observation(
            columns=list(self.dataframe.columns),
            null_counts=self.dataframe.isnull().sum().to_dict(),
            dtypes={col: str(dtype) for col, dtype in self.dataframe.dtypes.items()},
            target_schema_instructions=self.instructions,
            last_action_feedback=feedback
        )

    def step(self, action: Action):
        self.step_count += 1
        reward = 0.0
        terminated = False
        feedback = ""

        try:
            # ACTUAL PANDAS LOGIC (No Hacks)
            if action.tool == "fill_missing_values" and action.target_column in self.dataframe.columns:
                self.dataframe[action.target_column] = self.dataframe[action.target_column].fillna(action.new_value)
                reward = 0.5  # Organic partial credit!
                feedback = f"Success: Filled nulls in {action.target_column} with {action.new_value}."
                
            elif action.tool == "change_data_type" and action.target_column in self.dataframe.columns:
                self.dataframe[action.target_column] = self.dataframe[action.target_column].astype(action.new_value)
                reward = 0.5
                feedback = f"Success: Changed {action.target_column} to type {action.new_value}."
                
            elif action.tool == "drop_missing_rows" and action.target_column in self.dataframe.columns:
                self.dataframe = self.dataframe.dropna(subset=[action.target_column])
                reward = 0.5
                feedback = f"Success: Dropped missing rows in {action.target_column}."
                
            elif action.tool == "submit_final_dataset":
                terminated = True
                feedback = "Dataset submitted."
            else:
                feedback = f"Error: Tool '{action.tool}' failed or column not found."

        except Exception as e:
            feedback = f"Action crashed the environment: {str(e)}"

        # Prevent infinite loops on the server
        if self.step_count >= 10:
            terminated = True

        return self._get_obs(feedback), reward, terminated, False, {}