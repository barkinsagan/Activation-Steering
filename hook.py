import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Union
from collections import defaultdict

# ==============================================================================
# MODULE: Hook.py (Activation & Steering Management) - V1.1
# ==============================================================================

# --- 1. ActivationHook Class (Manager) ---
class ActivationHook:
    """
    Manages storage of activations (Read) and steering vectors (Write).
    """

    def __init__(self):
        # Storage for captured data (READ)
        self.activations: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.enabled = True

        # Storage for steering vectors (WRITE)
        # Format: {layer_name: tensor_vector}
        self.steering_vectors: Dict[str, torch.Tensor] = {}
        # Format: {layer_name: coefficient_float}
        self.steering_coeffs: Dict[str, float] = {}

    # --- Read Methods ---
    def clear(self):
        """Clear stored activations."""
        self.activations.clear()

    def remove_hooks(self):
        """Remove all registered hooks from the model."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def get_activation(self, layer_name: str) -> List[torch.Tensor]:
        return self.activations.get(layer_name, [])

    def get_all_activations(self) -> Dict[str, List[torch.Tensor]]:
        return dict(self.activations)

    # --- Write (Steering) Methods ---
    def set_steering_vector(
        self,
        layer_name: str,
        vector: torch.Tensor,
        coeff: Union[float, torch.Tensor] = 1.0,
    ):
        """
        Register a steering vector for a specific layer.

        ``coeff`` can be either a Python scalar (applied uniformly to the whole
        batch) or a 1-D Tensor of shape ``[B]`` (per-batch-row coefficient, used
        for coef-batched inference).

        NOTE: Vector is stored on CPU to save VRAM until inference time. Tensor
        coefs are also stored on CPU and moved to the activation's device/dtype
        on each forward pass.
        """
        self.steering_vectors[layer_name] = vector.detach().cpu()
        if isinstance(coeff, torch.Tensor):
            self.steering_coeffs[layer_name] = coeff.detach().cpu().float()
        else:
            self.steering_coeffs[layer_name] = float(coeff)

    def reset_steering(self):
        """Remove all steering vectors; return model to base behavior."""
        self.steering_vectors.clear()
        self.steering_coeffs.clear()


# --- 2. ModelWithHooks Class (Wrapper) ---
class ModelWithHooks:
    """Wrapper class to add Read/Write hook functionality to any PyTorch model."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.hook_manager = ActivationHook()

    def register_forward_hook(self, layer_name: str, module: nn.Module):
        """
        Register a forward hook that can both CAPTURE and STEER.
        """
        def hook_fn(module, input, output):
            # --- A. STEERING LOGIC (Write) ---
            # Check if we have a steering vector for this layer
            if layer_name in self.hook_manager.steering_vectors:
                steer_vec = self.hook_manager.steering_vectors[layer_name]
                coeff = self.hook_manager.steering_coeffs[layer_name]

                # Handle Tuple Outputs (Common in Transformer Blocks)
                is_tuple = isinstance(output, tuple)
                tensor_to_modify = output[0] if is_tuple else output

                # Move steer vector to correct device/dtype
                steer_vec = steer_vec.to(tensor_to_modify.device, dtype=tensor_to_modify.dtype)

                # Apply Intervention: h' = h + (coeff * v)
                # coeff may be a Python scalar (uniform over batch) OR a Tensor of
                # shape [B] for per-batch-row coefficients (coef-batched inference).
                if isinstance(coeff, torch.Tensor):
                    coeff_t = coeff.to(tensor_to_modify.device, dtype=tensor_to_modify.dtype)
                    # Broadcast [B] → [B,1,1] against [B, seq, hidden]
                    modified_tensor = tensor_to_modify + coeff_t.view(-1, 1, 1) * steer_vec
                else:
                    modified_tensor = tensor_to_modify + (coeff * steer_vec)

                # Repackage if it was a tuple
                if is_tuple:
                    output = (modified_tensor,) + output[1:]
                else:
                    output = modified_tensor

            # --- B. CAPTURE LOGIC (Read) ---
            # We capture *after* steering to see the modified state
            if self.hook_manager.enabled:
                if isinstance(output, torch.Tensor):
                    self.hook_manager.activations[layer_name].append(output.detach().cpu())
                elif isinstance(output, tuple):
                    # Recursively detach elements
                    self.hook_manager.activations[layer_name].append(
                        tuple(o.detach().cpu() if isinstance(o, torch.Tensor) else o for o in output)
                    )
            
            # --- C. RETURN ---
            # Crucial: Must return the modified output for steering to take effect
            return output

        handle = module.register_forward_hook(hook_fn)
        self.hook_manager.hooks.append(handle)

    def register_hooks_by_type(self, module_type: type):
        for name, module in self.model.named_modules():
            if isinstance(module, module_type):
                self.register_forward_hook(name, module)

    def register_hooks_on_layers(self, layer_names: List[str]):
        model_dict = dict(self.model.named_modules())
        for layer_name in layer_names:
            if layer_name in model_dict:
                self.register_forward_hook(layer_name, model_dict[layer_name])
            else:
                print(f"Warning: Layer '{layer_name}' not found in model")

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def clear_activations(self):
        self.hook_manager.clear()
        
    def reset_steering(self):
        self.hook_manager.reset_steering()

    def set_steering(
        self,
        layer_name: str,
        vector: torch.Tensor,
        coeff: Union[float, torch.Tensor] = 1.0,
    ):
        self.hook_manager.set_steering_vector(layer_name, vector, coeff)

    def get_activations(self, layer_name: Optional[str] = None):
        if layer_name:
            return self.hook_manager.get_activation(layer_name)
        return self.hook_manager.get_all_activations()


# --- 3. Helper Functions ---
def query_model_with_hooks(model_with_hooks: ModelWithHooks,
                           input_data: torch.Tensor) -> torch.Tensor:
    """Query model with hook tracking (and steering if configured)."""
    model_with_hooks.clear_activations()
    output = model_with_hooks(input_data)
    return output

def complete_with_hooks(model_with_hooks: ModelWithHooks,
                        input_data: torch.Tensor,
                        layer_to_analyze: Optional[str] = None) -> Dict[str, Any]:
    """Complete inference and return results with activations."""
    output = query_model_with_hooks(model_with_hooks, input_data)

    if layer_to_analyze:
        activations = model_with_hooks.get_activations(layer_to_analyze)
    else:
        activations = model_with_hooks.get_activations()

    return {
        'output': output,
        'activations': activations
    }

if __name__ == "__main__":
    print("hook.py module definitions complete.")
