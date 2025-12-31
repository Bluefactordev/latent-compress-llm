"""
Compressore Perceiver-style per compressione latente
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return self.weight * x / (norm + self.eps)


class SwiGLU(nn.Module):
    """SwiGLU activation"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.linear_gate = nn.Linear(dim, dim)
        self.linear = nn.Linear(dim, dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.linear_gate(x))
        return gate * self.linear(x)


class CrossAttentionBlock(nn.Module):
    """Cross-attention: Q da latents, K/V da teacher hidden states (con time-bucket embeddings)"""
    
    def __init__(self, d_lat: int, d_teacher: int, n_heads: int = 8, num_time_buckets: int = 8):
        super().__init__()
        self.d_lat = d_lat
        self.d_teacher = d_teacher
        self.n_heads = n_heads
        self.head_dim = d_lat // n_heads
        self.num_time_buckets = num_time_buckets
        
        assert d_lat % n_heads == 0, "d_lat deve essere divisibile per n_heads"
        
        # Proiezione teacher hidden states a d_lat
        self.teacher_proj = nn.Linear(d_teacher, d_lat)
        
        # Time-bucket embeddings (Fase 3)
        self.time_bucket_embedding = nn.Embedding(num_time_buckets, d_lat)
        
        # Q da latents
        self.q_proj = nn.Linear(d_lat, d_lat)
        # K, V da teacher states proiettati
        self.k_proj = nn.Linear(d_lat, d_lat)
        self.v_proj = nn.Linear(d_lat, d_lat)
        self.out_proj = nn.Linear(d_lat, d_lat)
        
        self.norm = RMSNorm(d_lat)
    
    def forward(self, latents: torch.Tensor, teacher_states: torch.Tensor, 
                token_positions: Optional[torch.Tensor] = None, 
                chunk_len: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            latents: [N_lat, d_lat]
            teacher_states: [seq_len, d_teacher]
            token_positions: [seq_len] posizioni token nel chunk (opzionale)
            chunk_len: lunghezza chunk per time-bucket (opzionale)
        
        Returns:
            [N_lat, d_lat]
        """
        # Proietta teacher states
        teacher_proj = self.teacher_proj(teacher_states)  # [seq_len, d_lat]
        
        # Aggiungi time-bucket embeddings (Fase 3)
        if token_positions is not None and chunk_len is not None:
            seq_len = teacher_states.shape[0]
            time_buckets = torch.floor(token_positions / chunk_len * self.num_time_buckets).long()
            time_buckets = torch.clamp(time_buckets, 0, self.num_time_buckets - 1)
            time_emb = self.time_bucket_embedding(time_buckets)  # [seq_len, d_lat]
            teacher_proj = teacher_proj + time_emb
        
        # Q da latents
        q = self.q_proj(latents)  # [N_lat, d_lat]
        # K, V da teacher
        k = self.k_proj(teacher_proj)  # [seq_len, d_lat]
        v = self.v_proj(teacher_proj)  # [seq_len, d_lat]
        
        # Reshape per multi-head
        N_lat = latents.shape[0]
        seq_len = teacher_states.shape[0]
        
        q = q.view(N_lat, self.n_heads, self.head_dim)  # [N_lat, n_heads, head_dim]
        k = k.view(seq_len, self.n_heads, self.head_dim)  # [seq_len, n_heads, head_dim]
        v = v.view(seq_len, self.n_heads, self.head_dim)  # [seq_len, n_heads, head_dim]
        
        # Transpose per attention
        q = q.transpose(0, 1)  # [n_heads, N_lat, head_dim]
        k = k.transpose(0, 1)  # [n_heads, seq_len, head_dim]
        v = v.transpose(0, 1)  # [n_heads, seq_len, head_dim]
        
        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)  # [n_heads, N_lat, seq_len]
        attn_output = torch.matmul(attn_weights, v)  # [n_heads, N_lat, head_dim]
        
        # Concatena heads
        attn_output = attn_output.transpose(0, 1).contiguous()  # [N_lat, n_heads, head_dim]
        attn_output = attn_output.view(N_lat, self.d_lat)  # [N_lat, d_lat]
        
        # Output projection
        output = self.out_proj(attn_output)
        
        # Residual + norm
        output = self.norm(output + latents)
        
        return output


class SelfAttentionBlock(nn.Module):
    """Self-attention sui latents"""
    
    def __init__(self, d_lat: int, n_heads: int = 8):
        super().__init__()
        self.d_lat = d_lat
        self.n_heads = n_heads
        self.head_dim = d_lat // n_heads
        
        self.q_proj = nn.Linear(d_lat, d_lat)
        self.k_proj = nn.Linear(d_lat, d_lat)
        self.v_proj = nn.Linear(d_lat, d_lat)
        self.out_proj = nn.Linear(d_lat, d_lat)
        
        self.norm = RMSNorm(d_lat)
    
    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latents: [N_lat, d_lat]
        
        Returns:
            [N_lat, d_lat]
        """
        N_lat = latents.shape[0]
        
        q = self.q_proj(latents)
        k = self.k_proj(latents)
        v = self.v_proj(latents)
        
        # Reshape
        q = q.view(N_lat, self.n_heads, self.head_dim).transpose(0, 1)
        k = k.view(N_lat, self.n_heads, self.head_dim).transpose(0, 1)
        v = v.view(N_lat, self.n_heads, self.head_dim).transpose(0, 1)
        
        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Concatena
        attn_output = attn_output.transpose(0, 1).contiguous().view(N_lat, self.d_lat)
        output = self.out_proj(attn_output)
        
        # Residual + norm
        output = self.norm(output + latents)
        
        return output


class CompressorBlock(nn.Module):
    """Blocco completo: cross-attn + self-attn + FF"""
    
    def __init__(self, d_lat: int, d_teacher: int, n_heads: int = 8, ff_dim: int = 2048, num_time_buckets: int = 8):
        super().__init__()
        self.cross_attn = CrossAttentionBlock(d_lat, d_teacher, n_heads, num_time_buckets)
        self.self_attn = SelfAttentionBlock(d_lat, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_lat, ff_dim),
            SwiGLU(ff_dim),
            nn.Linear(ff_dim, d_lat)
        )
        self.norm = RMSNorm(d_lat)
    
    def forward(self, latents: torch.Tensor, teacher_states: torch.Tensor,
                token_positions: Optional[torch.Tensor] = None,
                chunk_len: Optional[int] = None) -> torch.Tensor:
        # Cross-attention (con time-bucket embeddings)
        latents = self.cross_attn(latents, teacher_states, token_positions, chunk_len)
        # Self-attention
        latents = self.self_attn(latents)
        # Feed-forward
        ff_out = self.ff(latents)
        latents = self.norm(latents + ff_out)
        return latents


class GlobalResampler(nn.Module):
    """Second-stage resampler per latents globali"""
    
    def __init__(self, d_lat: int, n_global: int, n_heads: int = 8):
        super().__init__()
        self.n_global = n_global
        self.d_lat = d_lat
        
        # Latents globali learnable
        self.global_latents = nn.Parameter(torch.randn(n_global, d_lat) * 0.02)
        
        # Cross-attn da global a chunk latents
        self.cross_attn = CrossAttentionBlock(d_lat, d_lat, n_heads)
        self.self_attn = SelfAttentionBlock(d_lat, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_lat, d_lat * 4),
            SwiGLU(d_lat * 4),
            nn.Linear(d_lat * 4, d_lat)
        )
        self.norm = RMSNorm(d_lat)
    
    def forward(self, chunk_latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            chunk_latents: [num_chunks * N_lat, d_lat] (concatenati)
        
        Returns:
            [n_global, d_lat]
        """
        # Cross-attention da global latents a chunk latents
        global_lat = self.global_latents  # [n_global, d_lat]
        global_lat = self.cross_attn(global_lat, chunk_latents)
        global_lat = self.self_attn(global_lat)
        ff_out = self.ff(global_lat)
        global_lat = self.norm(global_lat + ff_out)
        return global_lat


class LatentCompressor(nn.Module):
    """Compressore principale con multi-layer support"""
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.n_latents = config["n_latents"]
        self.d_lat = config["d_lat"]
        self.d_teacher = config.get("d_teacher", 4096)  # Default per Qwen
        self.n_blocks = config.get("n_compressor_blocks", 2)
        self.n_heads = config.get("n_heads", 8)
        self.ff_dim = config.get("ff_dim", 2048)
        self.use_global_resampler = config.get("use_global_resampler", False)
        self.n_global = config.get("n_latents_global", 8192)
        self.num_time_buckets = config.get("num_time_buckets", 8)
        
        # Fase 2: Multi-layer support
        self.teacher_layers = config.get("teacher_layers", [-6, -18, -30])
        self.n_layers = len(self.teacher_layers)
        
        # Proiezioni separate per ogni layer
        self.layer_projections = nn.ModuleList([
            nn.Linear(self.d_teacher, self.d_lat)
            for _ in range(self.n_layers)
        ])
        
        # Proiezione finale dopo concat
        self.final_proj = nn.Linear(self.d_lat * self.n_layers, self.d_lat)
        
        # Latent tokens iniziali
        self.latent_tokens = nn.Parameter(
            torch.randn(self.n_latents, self.d_lat) * 0.02
        )
        
        # Embeddings posizionali (Fase 3)
        self.chunk_id_embedding = nn.Embedding(1000, self.d_lat)  # Max 1000 chunk
        self.latent_idx_embedding = nn.Embedding(self.n_latents, self.d_lat)
        
        # Blocchi compressore
        self.blocks = nn.ModuleList([
            CompressorBlock(self.d_lat, self.d_lat, self.n_heads, self.ff_dim, self.num_time_buckets)
            for _ in range(self.n_blocks)
        ])
        
        # Global resampler opzionale
        if self.use_global_resampler:
            self.global_resampler = GlobalResampler(self.d_lat, self.n_global, self.n_heads)
        else:
            self.global_resampler = None
    
    def forward(self, teacher_states_list: list, chunk_ids: Optional[list] = None,
                token_positions_list: Optional[list] = None) -> torch.Tensor:
        """
        Args:
            teacher_states_list: Lista di dict {layer_idx: [seq_len_i, d_teacher]} per ogni chunk
                                 O lista di [seq_len_i, d_teacher] se single-layer (backward compat)
            chunk_ids: Lista di ID chunk (opzionale)
            token_positions_list: Lista di [seq_len_i] posizioni token per time-bucket (opzionale)
        
        Returns:
            Latents finali: [total_latents, d_lat]
        """
        all_latents = []
        
        for chunk_idx, teacher_states in enumerate(teacher_states_list):
            # Fase 2: Gestione multi-layer
            if isinstance(teacher_states, dict):
                # Multi-layer: dict {layer_idx: tensor}
                layer_states = []
                for layer_proj, layer_key in zip(self.layer_projections, self.teacher_layers):
                    if layer_key in teacher_states:
                        layer_tensor = teacher_states[layer_key]  # [seq_len, d_teacher]
                        layer_proj_tensor = layer_proj(layer_tensor)  # [seq_len, d_lat]
                        layer_states.append(layer_proj_tensor)
                
                # Concatena layer
                if len(layer_states) > 1:
                    teacher_states_combined = torch.cat(layer_states, dim=-1)  # [seq_len, d_lat * n_layers]
                    teacher_states_proj = self.final_proj(teacher_states_combined)  # [seq_len, d_lat]
                else:
                    teacher_states_proj = layer_states[0] if layer_states else teacher_states
                
                seq_len = teacher_states_proj.shape[0]
                chunk_len = seq_len
            else:
                # Single-layer (backward compat): [seq_len, d_teacher]
                teacher_states_proj = self.layer_projections[0](teacher_states)  # [seq_len, d_lat]
                seq_len = teacher_states_proj.shape[0]
                chunk_len = seq_len
            
            # Inizializza latents per questo chunk
            latents = self.latent_tokens.clone()  # [N_lat, d_lat]
            
            # Aggiungi embeddings posizionali (Fase 3)
            if chunk_ids is not None:
                chunk_id = chunk_ids[chunk_idx]
            else:
                chunk_id = chunk_idx
            
            chunk_emb = self.chunk_id_embedding(torch.tensor(chunk_id, device=latents.device))
            latent_indices = torch.arange(self.n_latents, device=latents.device)
            latent_emb = self.latent_idx_embedding(latent_indices)
            
            latents = latents + chunk_emb.unsqueeze(0) + latent_emb
            
            # Token positions per time-bucket (Fase 3)
            if token_positions_list is not None and chunk_idx < len(token_positions_list):
                token_positions = token_positions_list[chunk_idx]
            else:
                # Default: posizioni sequenziali
                token_positions = torch.arange(seq_len, device=latents.device)
            
            # Applica blocchi compressore (con time-bucket embeddings)
            for block in self.blocks:
                latents = block(latents, teacher_states_proj, token_positions, chunk_len)
            
            all_latents.append(latents)
        
        # Concatena tutti i latents
        global_latents = torch.cat(all_latents, dim=0)  # [num_chunks * N_lat, d_lat]
        
        # Applica global resampler se richiesto
        if self.global_resampler is not None:
            global_latents = self.global_resampler(global_latents)
        
        return global_latents


