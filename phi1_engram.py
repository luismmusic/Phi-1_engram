from typing import List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import math
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, PhiConfig, PhiModel, PhiForCausalLM, PhiPreTrainedModel
from transformers.models.phi.modeling_phi import PhiDecoderLayer, PhiAttention, PhiMLP, PhiRotaryEmbedding
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from tokenizers import normalizers, Regex
from sympy import isprime

# =============================================================================
# 1. TOKENIZADOR COMPRIMIDO (CompressedTokenizer)
# Implementa la técnica de "Tokenizer Compression" descrita en la Sección 2.2 del paper.
# Su objetivo es maximizar la densidad semántica mediante una función suryectiva P: V -> V'.
# Colapsa múltiples IDs de tokens (que pueden ser semánticamente equivalentes pero
# representados distinto, ej: " Apple" vs "apple") en identificadores canónicos.
# Esto reduce el tamaño efectivo del vocabulario (aprox. 23% según el paper) y
# mitiga la fragmentación de la memoria en patrones de n-gramas.
# =============================================================================
class CompressedTokenizer:
    def __init__(
        self,
        tokenizer_name_or_path: str,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)

        SENTINEL = "\uE000"
        self.normalizer = normalizers.Sequence([
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), SENTINEL),
            normalizers.Strip(),
            normalizers.Replace(SENTINEL, " "),
        ])

        self.lookup_table, self.num_new_token = self._build_lookup_table()

    def __len__(self):
        return self.num_new_token

    def _build_lookup_table(self):
        old2new = {}
        key2new = {}
        new_tokens = []

        vocab_size = len(self.tokenizer)
        for tid in range(vocab_size):
            text = self.tokenizer.decode([tid], skip_special_tokens=False)

            if "\ufffd" in text:
                key = self.tokenizer.convert_ids_to_tokens(tid)
            else:
                norm = self.normalizer.normalize_str(text)
                key = norm if norm else text

            nid = key2new.get(key)
            if nid is None:
                nid = len(new_tokens)
                key2new[key] = nid
                new_tokens.append(key)
            old2new[tid] = nid

        lookup = np.empty(vocab_size, dtype=np.int64)
        for tid in range(vocab_size):
            lookup[tid] = old2new[tid]

        return lookup, len(new_tokens)

    def _compress(self, input_ids):
        arr = np.asarray(input_ids, dtype=np.int64)
        pos_mask = arr >= 0
        out = arr.copy()
        valid_ids = arr[pos_mask]
        out[pos_mask] = self.lookup_table[valid_ids]
        return out

    def __call__(self, input_ids):
        return self._compress(input_ids)

# Función auxiliar para encontrar números primos (necesario para el hashing)
def find_next_prime(start, seen_primes):
    candidate = start + 1
    while True:
        if isprime(candidate) and candidate not in seen_primes:
            return candidate
        candidate += 1

# =============================================================================
# 2. MAPEADOR DE HASHES (NgramHashMapping)
# Implementa el "Multi-Head Hashing" (Sección 2.2). Gestiona la extracción de n-gramas
# de sufijo y su mapeo a índices de tablas de embeddings mediante una función de
# hash determinista (multiplicative-XOR).
#
# Características técnicas:
# - Soporta múltiples órdenes de n-gramas (ej. 2-gramas, 3-gramas).
# - Utiliza K cabezales de hash por cada orden para mitigar colisiones.
# - El tamaño de cada tabla (M_n,k) es un número primo para mejorar la distribución.
# - Es determinista: depende únicamente de la secuencia de entrada de tokens.
# =============================================================================
class NgramHashMapping:
    def __init__(
        self,
        engram_vocab_size: List[int],
        max_ngram_size: int,
        n_embed_per_ngram: int,
        n_head_per_ngram: int,
        layer_ids: List[int],
        tokenizer_name_or_path: str,
        pad_id: int,
        seed: int,
    ):
        self.vocab_size_per_ngram = engram_vocab_size
        self.max_ngram_size = max_ngram_size
        self.n_embed_per_ngram = n_embed_per_ngram
        self.n_head_per_ngram = n_head_per_ngram
        self.pad_id = pad_id
        self.layer_ids = layer_ids

        self.compressed_tokenizer = CompressedTokenizer(
            tokenizer_name_or_path=tokenizer_name_or_path
        )
        self.tokenizer_vocab_size = len(self.compressed_tokenizer)
        if self.pad_id is not None:
            # We assume pad_id refers to original tokenizer, so we map it
            try:
                self.pad_id = int(self.compressed_tokenizer.lookup_table[self.pad_id])
            except IndexError:
                # If pad_id is out of range, we use a default
                self.pad_id = 0

        max_long = np.iinfo(np.int64).max
        M_max = int(max_long // self.tokenizer_vocab_size)
        half_bound = max(1, M_max // 2)
        PRIME_1 = 10007

        self.layer_multipliers = {}

        for layer_id in self.layer_ids:
            base_seed = int(seed + PRIME_1 * int(layer_id))
            g = np.random.default_rng(base_seed)
            r = g.integers(
                low=0,
                high=half_bound,
                size=(self.max_ngram_size,),
                dtype=np.int64
            )
            multipliers = r * 2 + 1
            self.layer_multipliers[layer_id] = multipliers

        self.vocab_size_across_layers = self.calculate_vocab_size_across_layers()

    def calculate_vocab_size_across_layers(self):
        seen_primes = set()
        vocab_size_across_layers = {}

        for layer_id in self.layer_ids:
            all_ngram_vocab_sizes = []
            for ngram in range(2, self.max_ngram_size + 1):
                current_ngram_heads_sizes = []

                vocab_size = self.vocab_size_per_ngram[ngram - 2]
                num_head = self.n_head_per_ngram
                current_prime_search_start = vocab_size - 1

                for _ in range(num_head):
                    found_prime = find_next_prime(
                        current_prime_search_start,
                        seen_primes
                    )
                    seen_primes.add(found_prime)
                    current_ngram_heads_sizes.append(found_prime)
                    current_prime_search_start = found_prime

                all_ngram_vocab_sizes.append(current_ngram_heads_sizes)
            vocab_size_across_layers[layer_id] = all_ngram_vocab_sizes

        return vocab_size_across_layers

    def _get_ngram_hashes(
        self,
        input_ids: np.ndarray,
        layer_id: int,
    ) -> np.ndarray:
        x = np.asarray(input_ids, dtype=np.int64)
        B, T = x.shape

        multipliers = self.layer_multipliers[layer_id]

        def shift_k(k: int) -> np.ndarray:
            if k == 0: return x
            shifted = np.pad(x, ((0, 0), (k, 0)),
                                mode='constant', constant_values=self.pad_id)[:, :T]
            return shifted

        base_shifts = [shift_k(k) for k in range(self.max_ngram_size)]

        all_hashes = []

        for n in range(2, self.max_ngram_size + 1):
            n_gram_index = n - 2
            tokens = base_shifts[:n]
            mix = (tokens[0] * multipliers[0])
            for k in range(1, n):
                mix = np.bitwise_xor(mix, tokens[k] * multipliers[k])
            num_heads_for_this_ngram = self.n_head_per_ngram
            head_vocab_sizes = self.vocab_size_across_layers[layer_id][n_gram_index]

            for j in range(num_heads_for_this_ngram):
                mod = int(head_vocab_sizes[j])
                head_hash = mix % mod
                all_hashes.append(head_hash.astype(np.int64, copy=False))

        return np.stack(all_hashes, axis=2)

    def hash(self, input_ids):
        # input_ids: list or np.ndarray or torch.Tensor
        if isinstance(input_ids, torch.Tensor):
            input_ids_np = input_ids.cpu().numpy()
        else:
            input_ids_np = np.asarray(input_ids)

        compressed_ids = self.compressed_tokenizer(input_ids_np)
        hash_ids_for_all_layers = {}
        for layer_id in self.layer_ids:
            hash_ids_for_all_layers[layer_id] = self._get_ngram_hashes(compressed_ids, layer_id=layer_id)
        return hash_ids_for_all_layers

# =============================================================================
# 3. EMBEDDINGS MULTI-CABEZAL (MultiHeadEmbedding)
# Almacén de "Static Priors". Implementa las tablas de embeddings E_{n,k} donde
# residen los parámetros de la memoria condicional.
# Utiliza offsets internos para indexar eficientemente múltiples cabezales de
# diferentes tamaños en una única estructura de almacenamiento.
# =============================================================================
class MultiHeadEmbedding(nn.Module):
    def __init__(self, list_of_N: List[int], D: int):
        super().__init__()
        self.num_heads = len(list_of_N)
        self.embedding_dim = D

        offsets = [0]
        for n in list_of_N[:-1]:
            offsets.append(offsets[-1] + n)

        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))

        total_N = sum(list_of_N)
        self.embedding = nn.Embedding(num_embeddings=total_N, embedding_dim=D)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [B, L, num_heads]
        # self.offsets: [num_heads]
        # We need to add offsets to each head's indices
        shifted_input_ids = input_ids + self.offsets
        output = self.embedding(shifted_input_ids)
        # output: [B, L, num_heads, D]
        return output

# =============================================================================
# 4. CONVOLUCIÓN CORTA (ShortConv)
# Implementa la "Short Depthwise Causal Convolution" (Ecuación 5).
# Expande el campo receptivo de la memoria recuperada y añade no-linealidad.
#
# Detalles técnicos:
# - Kernel size (w) típicamente = 4.
# - Dilatación (δ) ajustada al orden máximo del n-grama.
# - Utiliza RMSNorm y activación SiLU.
# - Es causal: los resultados en el tiempo 't' solo dependen de t y su pasado.
# - Incluye manejo de caché para mantener la consistencia en generación incremental.
# =============================================================================
class ShortConv(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        kernel_size: int = 4,
        dilation: int = 1,
        norm_eps: float = 1e-5,
        hc_mult: int = 1,
        activation: bool = True,
    ):
        super().__init__()
        self.hc_mult = hc_mult
        self.activation = activation

        total_channels = hidden_size * hc_mult
        self.conv = nn.Conv1d(
            in_channels=total_channels,
            out_channels=total_channels,
            kernel_size=kernel_size,
            groups=total_channels,
            bias=False,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )

        self.norms = nn.ModuleList([
            nn.RMSNorm(hidden_size, eps=norm_eps)
            for _ in range(hc_mult)
        ])

        if self.activation:
            self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor, cache: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input:  (B,L,HC_MULT,D)
        Cache:  (B,HC_MULT*D,L_cache)
        Output: (B,L,HC_MULT,D), New_Cache
        """
        B, T, G, C = x.shape

        assert G == self.hc_mult, f"Input groups {G} != hc_mult {self.hc_mult}"

        normed_chunks = []
        for i in range(G):
            chunk = x[:, :, i, :]
            normed_chunks.append(self.norms[i](chunk))

        x_norm = torch.cat(normed_chunks, dim=-1) # (B, L, G*C)
        x_bct = x_norm.transpose(1, 2) # (B, G*C, L)

        if cache is not None:
            # Concatenate cache for causal convolution
            conv_input = torch.cat([cache, x_bct], dim=-1)
        else:
            # First pass: pad at the beginning
            conv_input = x_bct

        y_bct = self.conv(conv_input)
        # We want the output corresponding to the current T steps
        # If cache was used, the convolution output for the new steps is at the end
        if cache is not None:
            y_bct = y_bct[..., cache.shape[-1] : cache.shape[-1] + T]
        else:
            y_bct = y_bct[..., :T]

        if self.activation:
            y_bct = self.act_fn(y_bct)
        y = y_bct.transpose(1, 2).view(B, T, G, C).contiguous()

        # Update cache (we only need the last few tokens)
        max_cache_len = (self.conv.kernel_size[0] - 1) * self.conv.dilation[0]
        new_cache = conv_input[..., -max_cache_len:]

        return y, new_cache

# =============================================================================
# 5. MÓDULO PHI-ENGRAM (PhiEngram)
# El bloque de integración principal (Context-aware Gating + Fusion).
# Realiza el producto punto entre el estado oculto (Query) y la memoria
# recuperada (Key) para calcular un escalar de gating α_t (Ecuaciones 3 y 4).
#
# Lógica de Fusión:
# 1. Recupera embeddings estáticos e_t.
# 2. Calcula proyecciones k_t (Key) y v_t (Value).
# 3. Aplica gating: α_t = sigmoid( (RMSNorm(h_t) · RMSNorm(k_t)) / sqrt(d) ).
# 4. Modula el valor: v_tilde_t = α_t * v_t.
# 5. Aplica ShortConv y conexión residual.
# =============================================================================
class PhiEngram(nn.Module):
    def __init__(self, config, layer_id, vocab_size_across_layers):
        super().__init__()
        self.layer_id = layer_id
        self.config = config

        # Engram specific parameters (assumed to be in config)
        max_ngram_size = getattr(config, "max_ngram_size", 3)
        n_embed_per_ngram = getattr(config, "n_embed_per_ngram", 512)
        n_head_per_ngram = getattr(config, "n_head_per_ngram", 8)
        kernel_size = getattr(config, "engram_kernel_size", 4)
        hc_mult = getattr(config, "hc_mult", 1)
        hidden_size = config.hidden_size

        list_of_N = [x for y in vocab_size_across_layers[self.layer_id] for x in y]
        self.multi_head_embedding = MultiHeadEmbedding(
            list_of_N=list_of_N,
            D=n_embed_per_ngram // n_head_per_ngram,
        )

        self.short_conv = ShortConv(
            hidden_size=hidden_size,
            kernel_size=kernel_size,
            dilation=max_ngram_size,
            hc_mult=hc_mult,
        )

        engram_hidden_size = (max_ngram_size - 1) * n_embed_per_ngram
        self.value_proj = nn.Linear(engram_hidden_size, hidden_size)
        self.key_projs = nn.ModuleList(
            [nn.Linear(engram_hidden_size, hidden_size) for _ in range(hc_mult)]
        )
        self.norm1 = nn.ModuleList([nn.RMSNorm(hidden_size) for _ in range(hc_mult)])
        self.norm2 = nn.ModuleList([nn.RMSNorm(hidden_size) for _ in range(hc_mult)])

    def forward(self, hidden_states: torch.Tensor, hash_input_ids: torch.Tensor, cache: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        hidden_states: [B, L, D] or [B, L, HC_MULT, D]
        hash_input_ids: [B, L, num_heads]
        """
        # Ensure we have the HC_MULT dimension
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.unsqueeze(2) # [B, L, 1, D]

        G = hidden_states.shape[2]

        # embeddings: [B, L, num_heads, d] -> flatten to [B, L, num_heads * d]
        embeddings = self.multi_head_embedding(hash_input_ids).flatten(start_dim=-2)

        gates = []
        for hc_idx in range(G):
            key = self.key_projs[hc_idx](embeddings)
            normed_key = self.norm1[hc_idx](key)
            query = hidden_states[:, :, hc_idx, :]
            normed_query = self.norm2[hc_idx](query)

            # Gating calculation
            gate = (normed_key * normed_query).sum(dim=-1) / math.sqrt(self.config.hidden_size)
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gate = gate.sigmoid().unsqueeze(-1)
            gates.append(gate)

        gates = torch.stack(gates, dim=2) # [B, L, G, 1]

        value = gates * self.value_proj(embeddings).unsqueeze(2) # [B, L, G, D]

        # Apply ShortConv with cache
        conv_output, new_cache = self.short_conv(value, cache)
        output = value + conv_output

        # If input was 3D, return 3D
        if G == 1:
            output = output.squeeze(2)

        return output, new_cache

# =============================================================================
# 6. CONFIGURACIÓN (PhiEngramConfig)
# Extiende la configuración oficial de Microsoft Phi-1 para incluir los hiper-parámetros
# específicos de Engram.
#
# Parámetros destacados:
# - engram_layer_ids: Índices de las capas donde se inyecta el módulo Engram.
# - engram_vocab_size: Lista con el tamaño de las tablas de hash por orden de n-grama.
# - max_ngram_size: Longitud máxima de las secuencias de tokens recordadas.
# - n_embed_per_ngram: Dimensión del vector de memoria recuperado.
# =============================================================================
class PhiEngramConfig(PhiConfig):
    model_type = "phi_engram"

    def __init__(
        self,
        engram_vocab_size: List[int] = [51200*5, 51200*5],
        max_ngram_size: int = 3,
        n_embed_per_ngram: int = 512,
        n_head_per_ngram: int = 8,
        engram_layer_ids: List[int] = [1, 15],
        tokenizer_name_or_path: str = "microsoft/phi-1",
        engram_seed: int = 0,
        engram_kernel_size: int = 4,
        hc_mult: int = 1,
        pad_token_id: int = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )
        self.engram_vocab_size = engram_vocab_size
        self.max_ngram_size = max_ngram_size
        self.n_embed_per_ngram = n_embed_per_ngram
        self.n_head_per_ngram = n_head_per_ngram
        self.engram_layer_ids = engram_layer_ids
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.engram_seed = engram_seed
        self.engram_kernel_size = engram_kernel_size
        self.hc_mult = hc_mult

# =============================================================================
# 7. CAPA DEL DECODIFICADOR (PhiEngramDecoderLayer)
# Implementa la integración de "Multi-branch Architecture" (Sección 2.4).
# Inyecta el módulo PhiEngram de forma paralela a los componentes de atención (MLA)
# y FFN, manteniendo la conexión residual. Gestiona el acceso al caché de convolución.
# =============================================================================
class PhiEngramDecoderLayer(PhiDecoderLayer):
    def __init__(self, config: PhiEngramConfig, layer_idx: int, vocab_size_across_layers: dict):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        self.engram = None
        if layer_idx in config.engram_layer_ids:
            self.engram = PhiEngram(config, layer_idx, vocab_size_across_layers)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        hash_input_ids: Optional[torch.LongTensor] = None, # Changed to hash_input_ids
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Engram integration
        if self.engram is not None and hash_input_ids is not None:
            # Handle ShortConv cache
            engram_cache = None
            if use_cache and past_key_values is not None:
                if not hasattr(past_key_values, "engram_conv_cache"):
                    past_key_values.engram_conv_cache = {}
                engram_cache = past_key_values.engram_conv_cache.get(self.layer_idx)

            engram_output, new_engram_cache = self.engram(hidden_states, hash_input_ids, engram_cache)

            if use_cache and past_key_values is not None:
                past_key_values.engram_conv_cache[self.layer_idx] = new_engram_cache
        else:
            engram_output = 0

        # Self Attention
        attn_outputs, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        attn_outputs = self.resid_dropout(attn_outputs)

        feed_forward_hidden_states = self.resid_dropout(self.mlp(hidden_states))

        hidden_states = attn_outputs + feed_forward_hidden_states + engram_output + residual
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs

# =============================================================================
# 8. CUERPO DEL MODELO (PhiEngramModel)
# Coordina la ejecución global. Implementa la optimización de rendimiento crítica:
# calcula los hashes de n-gramas una sola vez para toda la secuencia y los
# distribuye a las capas interesadas, minimizando transferencias CPU-GPU.
#
# Soporte Incremental:
# Gestiona 'engram_tokens' en el objeto past_key_values para permitir el cálculo
# correcto de n-gramas durante la generación token por token (model.generate()).
# =============================================================================
class PhiEngramModel(PhiModel):
    config_class = PhiEngramConfig

    def __init__(self, config: PhiEngramConfig):
        # Llamamos al constructor de la clase base PreTrainedModel directamente
        # para evitar que PhiModel cree capas duplicadas en RAM.
        PhiPreTrainedModel.__init__(self, config)

        # Engram specific parameters
        engram_vocab_size = getattr(config, "engram_vocab_size", [51200*5, 51200*5])
        max_ngram_size = getattr(config, "max_ngram_size", 3)
        n_embed_per_ngram = getattr(config, "n_embed_per_ngram", 512)
        n_head_per_ngram = getattr(config, "n_head_per_ngram", 8)
        layer_ids = getattr(config, "engram_layer_ids", [1, 15])
        tokenizer_name_or_path = getattr(config, "tokenizer_name_or_path", "microsoft/phi-1")
        pad_id = getattr(config, "pad_token_id", None)
        if pad_id is None:
            pad_id = 50256 # Default for Phi-1
        seed = getattr(config, "engram_seed", 0)

        self.hash_mapping = NgramHashMapping(
            engram_vocab_size=engram_vocab_size,
            max_ngram_size=max_ngram_size,
            n_embed_per_ngram=n_embed_per_ngram,
            n_head_per_ngram=n_head_per_ngram,
            layer_ids=layer_ids,
            tokenizer_name_or_path=tokenizer_name_or_path,
            pad_id=pad_id,
            seed=seed,
        )

        # Inicialización de componentes base (manual para ahorrar memoria)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.rotary_emb = PhiRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.embed_dropout = nn.Dropout(config.embd_pdrop)
        self.final_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Inicializamos las capas de Engram directamente
        self.layers = nn.ModuleList(
            [PhiEngramDecoderLayer(config, layer_idx, self.hash_mapping.vocab_size_across_layers)
             for layer_idx in range(config.num_hidden_layers)]
        )

        # Inicialización final de pesos
        self.post_init()

    def _prepare_causal_mask(self, attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions, position_ids):
        """
        Versión robusta para crear la máscara causal, compatible con diferentes
        versiones de la biblioteca 'transformers'.
        """
        # Caso 1: La versión de transformers tiene _update_causal_mask como método de clase base
        if hasattr(self, "_update_causal_mask"):
            return self._update_causal_mask(
                attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
            )

        # Caso 2: Se usa la función independiente create_causal_mask (v4.4x y superiores)
        # Intentamos varias firmas para maximizar la compatibilidad con diferentes versiones de transformers
        try:
            # Firma completa con nombres de argumentos (v5.x / v4.4x recientes)
            return create_causal_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
        except TypeError:
            try:
                # Firma sin nombres de argumentos (posicional)
                return create_causal_mask(
                    self.config,
                    inputs_embeds,
                    attention_mask,
                    cache_position,
                    past_key_values,
                    position_ids,
                )
            except TypeError:
                try:
                    # Firma sin position_ids
                    return create_causal_mask(
                        self.config,
                        inputs_embeds,
                        attention_mask,
                        cache_position,
                        past_key_values,
                    )
                except:
                    # Último recurso: intentar sin config si es una versión muy antigua/específica
                    return create_causal_mask(
                        inputs_embeds,
                        attention_mask,
                        cache_position,
                        past_key_values,
                    )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Usamos el método robusto para evitar errores de compatibilidad en Colab
        causal_mask = self._prepare_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions, position_ids
        )

        inputs_embeds = self.embed_dropout(inputs_embeds)
        hidden_states = inputs_embeds

        # create position embeddings
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        # =====================================================================
        # OPTIMIZACIÓN DE HASHING (Pre-calculado)
        # Calculamos los hashes de n-gramas una sola vez para toda la secuencia.
        # Esto evita recalcular lo mismo en cada capa que tenga un módulo Engram.
        # =====================================================================
        all_layer_hashes = {}
        if input_ids is not None:
            # Soporte para Generación Incremental (Auto-regresiva):
            # Mantenemos un pequeño historial de tokens ('engram_tokens') en
            # past_key_values para poder calcular los n-gramas de sufijo cuando
            # solo nos llega 1 token nuevo.
            if use_cache and past_key_values is not None:
                if not hasattr(past_key_values, "engram_tokens"):
                    # Initial pass: we use the whole input_ids
                    effective_input_ids = input_ids
                    # But we store only the suffix needed for future n-grams
                    past_key_values.engram_tokens = input_ids[:, -self.hash_mapping.max_ngram_size:]
                else:
                    # Incremental pass: we concatenate history and current token(s)
                    effective_input_ids = torch.cat([past_key_values.engram_tokens, input_ids], dim=-1)
                    # Update cache
                    past_key_values.engram_tokens = effective_input_ids[:, -self.hash_mapping.max_ngram_size:]
            else:
                effective_input_ids = input_ids

            hashes_dict = self.hash_mapping.hash(effective_input_ids)
            for layer_idx, hashes in hashes_dict.items():
                # We only need the hashes corresponding to the current input_ids sequence length
                if hashes.shape[1] > input_ids.shape[1]:
                    hashes = hashes[:, -input_ids.shape[1]:, :]
                all_layer_hashes[layer_idx] = torch.from_numpy(hashes).to(hidden_states.device)

        for i, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                hash_input_ids=all_layer_hashes.get(i), # Pass pre-calculated hashes
                **kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.final_layernorm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = past_key_values if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

# =============================================================================
# 9. MODELO FINAL (PhiEngramForCausalLM)
# Es la clase que usarás tú. Permite cargar el modelo y generar texto.
# =============================================================================
class PhiEngramForCausalLM(PhiForCausalLM):
    config_class = PhiEngramConfig

    def __init__(self, config: PhiEngramConfig):
        # Saltamos el constructor de PhiForCausalLM para evitar la creación de
        # un PhiModel estándar redundante.
        PhiPreTrainedModel.__init__(self, config)

        self.model = PhiEngramModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=True)

        # Inicialización final de pesos
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Any:
        # Standard PhiForCausalLM forward but using our PhiEngramModel
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        from transformers.modeling_outputs import CausalLMOutputWithPast
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
