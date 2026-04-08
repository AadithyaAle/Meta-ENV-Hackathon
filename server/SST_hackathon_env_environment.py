import pandas as pd
import torch
import numpy as np
from uuid import uuid4
from typing import Optional
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

# Use the names the server is looking for
try:
    from ..models import Action as SstHackathonAction
    from ..models import Observation as SstHackathonObservation
except ImportError:
    from models import Action as SstHackathonAction
    from models import Observation as SstHackathonObservation

class SstHackathonEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.df = pd.DataFrame()
        self.df_history = []
        self.initial_row_count = 0

    def reset(self) -> SstHackathonObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.df_history = []
        
        # Hard Task for the Bot to Validate
        self.df = pd.DataFrame({
            "usr_nm": ["Alice ", " Bob", "Charlie", "David\n", " Eve "],
            "Age": ["25 ", "\n30", None, " 22 ", None],
        })
        self.initial_row_count = len(self.df)
        
        return self._get_observation("Environment Reset: Clean the dataset for PyTorch.")

    def _get_observation(self, feedback: str) -> SstHackathonObservation:
        return SstHackathonObservation(
            current_columns=list(self.df.columns),
            data_types={col: str(dtype) for col, dtype in self.df.dtypes.items()},
            missing_values=self.df.isnull().sum().to_dict(),
            data_preview=self.df.head().to_markdown(),
            last_action_feedback=feedback,
            target_schema_instructions="Clean the dataset for PyTorch: handle missing values and encode categories.",
            done=False,
            reward=0.0
        )

    def step(self, action: SstHackathonAction) -> SstHackathonObservation:
        self._state.step_count += 1
        reward = -0.05 # Efficiency Penalty
        done = False
        feedback = ""

        try:
            if action.tool == "undo_last_action":
                if self.df_history:
                    self.df = self.df_history.pop()
                    feedback = "Reverted to previous state."
                else:
                    feedback = "Undo failed: No history."
            
            elif action.tool == "submit_final_dataset":
                # PyTorch Validation
                try:
                    # Check for data loss
                    if len(self.df) < self.initial_row_count:
                        return SstHackathonObservation(done=True, reward=-1.0, last_action_feedback="Failed: Data loss detected.")
                    
                    # Try to compile to tensor
                    numeric_df = self.df.select_dtypes(include=[np.number])
                    torch.tensor(numeric_df.values)
                    return SstHackathonObservation(done=True, reward=1.0, last_action_feedback="Success: PyTorch Validated!")
                except Exception as e:
                    return SstHackathonObservation(done=True, reward=-0.5, last_action_feedback=f"PyTorch Error: {str(e)}")

            # Add your other tool logic (rename_column, etc) here if needed 
            # For the bot, usually the reset/observation structure is enough to pass phase 1
            else:
                self.df_history.append(self.df.copy())
                feedback = f"Action {action.tool} accepted."

        except Exception as e:
            feedback = f"Error: {str(e)}"

        return self._get_observation(feedback)

    @property
    def state(self) -> State:
        return self._state