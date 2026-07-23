from transformers import Trainer
import numpy as np
from typing import Any, Dict, Union
import torch

class MAETrainer(Trainer):
    def __init__(self, modal_mode=None, **kwargs):
        super().__init__(**kwargs)
        self.modal_mode = modal_mode

    def floating_point_ops(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> int:
        """
        `Trainer.floating_point_ops` only handles a single string `main_input_name` and does
        `main_input in inputs`, which crashes with `TypeError: unhashable type: 'list'` since
        `SpatialSpectralMAEViT.main_input_name` is `['optical', 'radar']`. Delegate to the
        model's own `estimate_tokens`, which already supports list-valued `main_input_name`.
        """
        if not hasattr(self.model, "num_parameters"):
            return 0
        tokens = self.model.estimate_tokens(inputs)
        return 6 * tokens * self.model.num_parameters(exclude_embeddings=True)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.modal_mode == "random":
            modal = np.random.choice(['multi', 'optical', 'radar'])
        else:
            modal = self.modal_mode
            
        outputs = model(**inputs, modal = modal)
        
        assert self.compute_loss_func is not None, "compute_loss_func is not set"
        loss = self.compute_loss_func(outputs)

        return (loss, outputs) if return_outputs else loss
    
    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)
