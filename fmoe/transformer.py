r"""
Adaption to act as the MLP layer using an MoE MLP layer in transformer.
"""
import torch
import torch.nn as nn
import tree
from .layers import FMoE, _fmoe_general_global_forward, ensure_comm, AllGather, Slice
from .linear import FMoELinear
from .fastermoe.config import switch_from_env


fmoe_faster_schedule = False
if switch_from_env('FMOE_FASTER_SCHEDULE_ENABLE', False):
    fmoe_faster_schedule = True
    from .fastermoe.schedule import _fmoe_general_global_forward


class _Expert(nn.Module):
    r"""
    An expert using 2 FMoELinear modules to speed up the computation of experts
    within one worker.
    """

    def __init__(self, num_expert, d_model, d_hidden, activation, rank=0):
        super().__init__()
        self.htoh4 = FMoELinear(num_expert, d_model, d_hidden, bias=True, rank=rank)
        self.h4toh = FMoELinear(num_expert, d_hidden, d_model, bias=True, rank=rank)
        self.activation = activation

    def forward(self, inp, fwd_expert_count):
        r"""
        First expand input to 4h (the hidden size is variable, but is called h4
        for convenience). Then perform activation. Finally shirink back to h.
        """
        x = self.htoh4(inp, fwd_expert_count)
        x = self.activation(x)
        x = self.h4toh(x, fwd_expert_count)
        return x


class SharedExpert(nn.Module):
    r"""
    Shared expert, can be implemented with either nn.Linear or FMoELinear.
    """

    def __init__(self, d_model, d_hidden, activation, use_fmoe_linear=False):
        super().__init__()
        self.use_fmoe_linear = use_fmoe_linear
        
        if use_fmoe_linear:
            # Use FMoELinear with num_expert=1 for consistency with other experts
            self.htoh4 = FMoELinear(1, d_model, d_hidden, bias=True, rank=0)
            self.h4toh = FMoELinear(1, d_hidden, d_model, bias=True, rank=0)
        else:
            # Use standard nn.Linear (current approach)
            self.htoh4 = nn.Linear(d_model, d_hidden)
            self.h4toh = nn.Linear(d_hidden, d_model)
        
        self.activation = activation

    def forward(self, inp):
        r"""
        Forward pass for shared expert.
        """
        if self.use_fmoe_linear:
            # For FMoELinear, we need to create a dummy expert count tensor
            batch_size = inp.shape[0]
            fwd_expert_count = torch.tensor([batch_size], device=inp.device, dtype=torch.long)
            
            x = self.htoh4(inp, fwd_expert_count)
            x = self.activation(x)
            x = self.h4toh(x, fwd_expert_count)
        else:
            x = self.htoh4(inp)
            x = self.activation(x)
            x = self.h4toh(x)
        return x



class FMoETransformerMLP(FMoE):
    r"""
    A complete MoE MLP module in a Transformer block.
    * `activation` is the activation function to be used in MLP in each expert.
    * `d_hidden` is the dimension of the MLP layer.
    """

    def __init__(
        self,
        num_expert=32,
        d_model=1024,
        d_hidden=4096,
        activation=torch.nn.GELU(),
        expert_dp_comm="none",
        expert_rank=0,
        expert_group=1,
        **kwargs
    ):
        # Calculate fine-grained dimensions based on expert_group
        self.expert_group = expert_group
        experts_per_group = num_expert // expert_group
        d_hidden_per_expert = d_hidden // experts_per_group if expert_group > 1 else d_hidden
        
        def one_expert(d_model):
            # Each expert uses fine-grained d_hidden dimensions
            return _Expert(1, d_model, d_hidden_per_expert, activation, rank=0)
        
        expert = one_expert
        super().__init__(num_expert=num_expert, d_model=d_model, expert=expert, **kwargs)
        self.mark_parallel_comm(expert_dp_comm)
        
        # SharedExpert always uses full d_hidden dimensions (reuses BART FFN weights)
        # You can set use_fmoe_linear=True if you want architectural consistency
        self.shared_expert = SharedExpert(d_model, d_hidden, activation, use_fmoe_linear=False)
        
        # Store dimensions for reference
        self.d_hidden = d_hidden  # Full hidden dimension  
        self.d_hidden_per_expert = d_hidden_per_expert  # Per-expert hidden dimension

    def forward(self, inp: torch.Tensor):
        r"""
        This module wraps up the FMoE module with reshape, residual and layer
        normalization.
        """
        original_shape = inp.shape
        inp_flat = inp.reshape(-1, self.d_model)
        routed_output = super().forward(inp_flat)
        shared_output = self.shared_expert(inp_flat)
        output = routed_output + shared_output
        return output.reshape(original_shape)


class SyntaxGuidedFMoETransformerMLP(FMoETransformerMLP):
    r"""
    A variant of FMoETransformerMLP that accepts separate inputs for the
    experts and for the gate. Use `forward(expert_input, gate_input)` where
    `expert_input` is the tensor fed to experts and `gate_input` is the tensor
    used to compute routing scores.
    """

    def forward(self, expert_input: torch.Tensor, gate_input: torch.Tensor):
        r"""
        This module wraps up the FMoE module with reshape, residual and layer
        normalization, but uses separate inputs for experts and gate.
        """
        original_shape = expert_input.shape
        expert_flat = expert_input.reshape(-1, self.d_model)
        gate_flat = gate_input.reshape(-1, self.d_model)

        # Ensure communication tensors are prepared (as in FMoE.forward)
        if self.world_size > 1:
            ensure_comm(expert_flat, self.moe_group)
            ensure_comm(gate_flat, self.moe_group)

        # Handle model slicing if enabled (mirror FMoE.forward behavior)
        if self.slice_size > 1:
            expert_flat = Slice.apply(expert_flat, self.slice_rank, self.slice_size, self.slice_group)
            gate_flat = Slice.apply(gate_flat, self.slice_rank, self.slice_size, self.slice_group)

        # Compute gate top-k indices and gate scores from gate_input
        gate_top_k_idx, gate_score = self.gate(gate_flat)

        # Perform the global MoE forward using the precomputed gate indices
        fwd = _fmoe_general_global_forward(
            expert_flat,
            gate_top_k_idx,
            self.expert_fn_single if fmoe_faster_schedule else self.expert_fn,
            self.num_expert,
            self.world_size,
            experts=self.experts
        )

        # Recover / reshape outputs similar to FMoE.forward
        if self.mask is not None and self.mask_dict is not None:
            def recover_func(tensor):
                dim = tensor.shape[-1]
                tensor = tensor.view(-1, self.top_k, dim)
                x = torch.zeros(
                    self.mask.shape[0],
                    self.top_k,
                    dim,
                    device=tensor.device,
                    dtype=tensor.dtype,
                )
                x[self.mask == 0] = tensor
                for k, v in self.mask_dict.items():
                    x[self.mask == k] = v
                return x

            moe_outp = tree.map_structure(recover_func, fwd)
        else:
            def view_func(tensor):
                dim = tensor.shape[-1]
                tensor = tensor.view(-1, self.top_k, dim)
                return tensor

            moe_outp = tree.map_structure(view_func, fwd)

        # Gate_score shape -> (B*T, 1, top_k) to bmm with moe_outp
        gate_score = gate_score.view(-1, 1, self.top_k)

        def bmm_func(tensor):
            dim = tensor.shape[-1]
            tensor = torch.bmm(gate_score, tensor).reshape(-1, dim)
            return tensor

        moe_outp = tree.map_structure(bmm_func, moe_outp)

        # If sliced, gather across slices
        if self.slice_size > 1:
            def all_gather_func(tensor):
                return AllGather.apply(tensor, self.slice_rank, self.slice_size, self.slice_group)

            moe_outp = tree.map_structure(all_gather_func, moe_outp)

        # Final sanity check and reshape back to original
        moe_outp_batch_size = tree.flatten(tree.map_structure(lambda tensor: tensor.shape[0], moe_outp))
        assert all([batch_size == moe_outp_batch_size[0] for batch_size in moe_outp_batch_size]), "MoE outputs must have the same batch size"

        # Add shared expert output
        shared_output = self.shared_expert(expert_flat)
        moe_outp = tree.map_structure(lambda x: x + shared_output, moe_outp)

        # Reshape returned tensors to original shape
        def final_view(tensor):
            return tensor.reshape(original_shape)

        return tree.map_structure(final_view, moe_outp)
