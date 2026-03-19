import torch
import numpy as np
from typing import List, Dict, Optional, Tuple

class DifferenceInMeansSteering:
    """
    Implements the Difference in Means approach for calculating steering vectors.

    Fixes included:
    1. Robust Masking: Handles padding correctly for both 'last' and 'mean' strategies.
    2. Memory Safety: Moves activations to CPU immediately.
    3. State Safety: Explicitly clears storage lists before capture to prevent double-counting.
    4. Deterministic: Sets model to eval mode and fixes random seeds.
    """

    def __init__(self,
                 model_with_hooks,
                 tokenizer,
                 target_layer: str,
                 token_position: str = "last",
                 seed: int = 42):
        """
        Args:
            model_with_hooks: ModelWithHooks instance
            tokenizer: HuggingFace tokenizer
            target_layer: Layer name (e.g., 'model.layers.10.mlp.down_proj')
            token_position: "last" (last real token) or "mean" (average of real tokens)
            seed: Random seed for reproducibility
        """
        self.model_with_hooks = model_with_hooks
        self.tokenizer = tokenizer
        self.target_layer = target_layer
        self.token_position = token_position
        self.seed = seed

        # Storage: List of tensors of shape [hidden_size]
        self.positive_activations: List[torch.Tensor] = []
        self.negative_activations: List[torch.Tensor] = []
        
        # Set tokenizer padding side for consistency
        self.tokenizer.padding_side = 'right'

    def _set_seed(self):
        """Set random seeds for reproducibility."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            # For deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _process_batch_activations(self,
                                   batch_activations: torch.Tensor,
                                   attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Extracts the relevant vector from the sequence based on strategy.
        Handles padding correctly using the attention mask.
        """
        # batch_activations: [batch, seq_len, hidden_size]
        # attention_mask:    [batch, seq_len] (1 for real, 0 for pad)

        # Ensure mask is on the same device
        attention_mask = attention_mask.to(batch_activations.device)

        if self.token_position == "last":
            # Find the index of the last non-padding token
            # sum(dim=1) gives count of real tokens. Subtract 1 for 0-based index.
            # We clamp to 0 just in case of an empty sequence (unlikely)
            last_indices = (attention_mask.sum(dim=1) - 1).clamp(min=0) # [batch]

            # Gather the vectors at those indices
            batch_indices = torch.arange(batch_activations.size(0), device=batch_activations.device)
            extracted = batch_activations[batch_indices, last_indices, :] # [batch, hidden]
            return extracted

        elif self.token_position == "mean":
            # Expand mask to [batch, seq, 1] for broadcasting
            mask_expanded = attention_mask.unsqueeze(-1).float()

            # Zero out padding positions
            masked_acts = batch_activations * mask_expanded

            # Sum across sequence
            sum_acts = masked_acts.sum(dim=1) # [batch, hidden]

            # Count real tokens (clamp to 1 to avoid divide by zero)
            token_counts = mask_expanded.sum(dim=1).clamp(min=1.0) # [batch, 1]

            # Compute mean
            mean_acts = sum_acts / token_counts
            return mean_acts

        else:
            raise ValueError(f"Unknown token_position strategy: {self.token_position}")

    def capture_activations(self,
                           prompts: List[str],
                           max_length: int = 512,
                           is_positive: bool = True) -> int:
        """
        Internal loop to process prompts and store activations.
        """
        dataset_type = "positive" if is_positive else "negative"
        print(f"📊 Capturing {dataset_type} activations from {len(prompts)} prompts...")

        # 0. Set deterministic behavior
        self._set_seed()
        self.model_with_hooks.model.eval()  # ← CRITICAL FIX

        # 1. Setup Model
        self.model_with_hooks.register_hooks_on_layers([self.target_layer])
        self.model_with_hooks.reset_steering()
        self.model_with_hooks.hook_manager.enable()

        # Select storage list (Pointer reference)
        storage = self.positive_activations if is_positive else self.negative_activations

        with torch.no_grad(): # Disable gradients to save memory
            for i, prompt in enumerate(prompts):
                # Tokenize with consistent padding
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=max_length,
                    truncation=True,
                    padding='max_length'  # ← CHANGED: consistent padding
                )

                input_ids = inputs['input_ids'].to(self.model_with_hooks.model.device)
                attention_mask = inputs['attention_mask'].to(self.model_with_hooks.model.device)

                # Clear previous hooks data
                self.model_with_hooks.clear_activations()

                # Forward pass (triggers hooks)
                _ = self.model_with_hooks(input_ids)

                # Retrieve raw activations
                raw_acts_list = self.model_with_hooks.get_activations(self.target_layer)

                if not raw_acts_list:
                    print(f"⚠️ Warning: No activations found for prompt {i}")
                    continue

                # Take the first element (batch)
                raw_act_tensor = raw_acts_list[0]

                # Process (Handle padding/masking)
                processed_acts = self._process_batch_activations(raw_act_tensor, attention_mask)

                # Move to CPU immediately
                processed_acts = processed_acts.detach().cpu()

                # Append individual vectors to storage
                for vec in processed_acts:
                    storage.append(vec)

                if (i + 1) % 20 == 0:
                    print(f"  ✓ Processed {i + 1}/{len(prompts)} {dataset_type} prompts")

        self.model_with_hooks.hook_manager.disable()
        print(f"✅ Captured {len(storage)} {dataset_type} vectors")
        return len(storage)

    def capture_positive_activations(self, prompts: List[str], max_length: int = 512) -> int:
        """Wrapper that ensures positive storage is clean before capture."""
        self.positive_activations = [] # Explicit reset
        return self.capture_activations(prompts, max_length, is_positive=True)

    def capture_negative_activations(self, prompts: List[str], max_length: int = 512) -> int:
        """Wrapper that ensures negative storage is clean before capture."""
        self.negative_activations = [] # Explicit reset
        return self.capture_activations(prompts, max_length, is_positive=False)

    def compute_steering_vector(self,
                               normalize: bool = False,
                               norm_type: str = "unit") -> torch.Tensor:
        """
        Computes v = Mean(Positive) - Mean(Negative)
        """
        print("\n🧮 Computing Difference in Means steering vector...")

        # Sanity Checks
        if not self.positive_activations:
            raise ValueError("No positive activations found. Run capture_positive_activations first.")
        if not self.negative_activations:
            raise ValueError("No negative activations found. Run capture_negative_activations first.")

        if self.positive_activations is self.negative_activations:
             raise ValueError("CRITICAL: Positive and Negative lists are the same object.")

        # Stack lists into tensors [N, hidden]
        try:
            pos_tensor = torch.stack(self.positive_activations)
            neg_tensor = torch.stack(self.negative_activations)
        except Exception as e:
            raise ValueError(f"Error stacking activations. Check tensor shapes. {e}")

        print(f"  Pos shape: {pos_tensor.shape}")
        print(f"  Neg shape: {neg_tensor.shape}")

        # Calculate Means
        mu_pos = pos_tensor.mean(dim=0)
        mu_neg = neg_tensor.mean(dim=0)

        # Difference
        steering_vector = mu_pos - mu_neg

        # Diagnostics
        cos_sim = torch.nn.functional.cosine_similarity(mu_pos.unsqueeze(0), mu_neg.unsqueeze(0)).item()
        print(f"  📐 Cosine similarity between means: {cos_sim:.4f}")
        
        # Additional diagnostic: check magnitude difference
        pos_norm = mu_pos.norm().item()
        neg_norm = mu_neg.norm().item()
        print(f"  📏 Pos mean norm: {pos_norm:.4f}, Neg mean norm: {neg_norm:.4f}")

        # Normalization
        if normalize:
            if norm_type == "unit":
                steering_vector = steering_vector / (steering_vector.norm() + 1e-8)
            elif norm_type == "std":
                combined = torch.cat([pos_tensor, neg_tensor], dim=0)
                std = combined.std(dim=0)
                steering_vector = steering_vector / (std + 1e-8)

        print(f"✅ Vector computed. Norm: {steering_vector.norm().item():.4f}")
        return steering_vector

    def apply_steering(self, vector: torch.Tensor, coefficient: float = 1.0):
        """Register the steering vector with the model."""
        print(f"🎯 Applying steering to {self.target_layer} with coeff {coefficient}")
        self.model_with_hooks.set_steering(self.target_layer, vector, coefficient)

    def reset_steering(self):
        """Disable steering."""
        self.model_with_hooks.reset_steering()
        print("🔄 Steering reset")

    def cleanup(self):
        """Free memory and remove hooks."""
        self.model_with_hooks.hook_manager.remove_hooks()
        self.positive_activations = []
        self.negative_activations = []
        print("🧹 Cleanup complete")
