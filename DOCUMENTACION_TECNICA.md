# Documentación Técnica: Arquitectura Phi-1 Engram

Esta documentación detalla la implementación técnica de la integración del módulo **DeepSeek Engram** en la arquitectura de **Microsoft Phi-1**, basada en el whitepaper *"Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models"* (Cheng et al., 2026).

---

## 1. El Concepto: Dualidad Lingüística

El modelo Engram parte de la premisa de que el lenguaje natural tiene dos componentes distintos:
1.  **Razonamiento Composicional**: Requiere cómputo profundo y dinámico (Atención y FFNs).
2.  **Recuperación de Conocimiento**: Muchos patrones (nombres de entidades, frases hechas) son locales y estáticos.

Los Transformers estándar simulan la memoria mediante cómputo, lo cual es ineficiente. Engram introduce una **memoria condicional** que permite realizar búsquedas de conocimiento en tiempo constante $O(1)$.

---

## 2. Componentes de la Arquitectura Engram

### 2.1 Tokenizer Compression
Para maximizar la densidad semántica, implementamos una función suryectiva $P: V \to V'$.
- **Problema**: Los tokenizadores estándar asignan IDs distintos a variaciones irrelevantes (ej: " Apple" vs "apple").
- **Solución**: Colapsamos tokens mediante normalización NFKC y paso a minúsculas.
- **Impacto**: Reduce el vocabulario efectivo en un ~23%, haciendo que los patrones de n-gramas sean más robustos y menos fragmentados.

### 2.2 Multi-Head Hashing
Mapeamos contextos locales (n-gramas de sufijo) a tablas de embeddings de forma determinista.
- **Hasing XOR-Multiplicativo**: Una función ligera que opera sobre los IDs de los tokens.
- **Cabezales Múltiples ($K$ heads)**: Para cada orden de n-grama (ej. 2-gramas, 3-gramas), usamos múltiples cabezales de hash para mitigar colisiones.
- **Tablas de Tamaño Primo**: Los tamaños de las tablas de embeddings se eligen como números primos para asegurar una distribución uniforme del hash.

### 2.3 Context-aware Gating (Fusión Adaptativa)
La memoria recuperada es estática; el gating la hace dinámica.
- **Query**: Proviene del *hidden state* actual ($h_t$), que ya tiene contexto global gracias a la atención.
- **Key/Value**: Provienen de la memoria recuperada ($e_t$).
- **Ecuación de Gating**:
  $$\alpha_t = \sigma\left(\frac{\text{RMSNorm}(h_t)^\top \cdot \text{RMSNorm}(W_K e_t)}{\sqrt{d}}\right)$$
- **Resultado**: Si la memoria recuperada contradice el contexto actual, el gate $\alpha_t$ se cierra (tiende a 0), eliminando el ruido.

### 2.4 Short Depthwise Causal Convolution
Para añadir no-linealidad y expandir el campo receptivo:
- Aplicamos una convolución causal 1D sobre los valores modulados.
- **Dilatación**: Se ajusta al orden máximo de n-grama para capturar dependencias temporales relevantes.
- **Consistencia**: Utilizamos un caché de convolución para que la generación incremental sea idéntica al procesamiento batch.

---

## 3. Integración con Microsoft Phi-1

### 3.1 Estrategia de Subclassing
En lugar de modificar el código fuente de `transformers`, extendemos sus clases:
- `PhiEngramConfig`: Hereda de `PhiConfig` y añade parámetros como `engram_layer_ids`.
- `PhiEngramModel`: Sobreescribe el `forward` para inyectar la lógica de hashing.
- `PhiEngramForCausalLM`: El envoltorio final que gestiona la pérdida (loss) y los logits.

### 3.2 Optimización de Rendimiento
En una implementación ingenua, el hashing se calcularía en cada capa. En nuestra versión:
1.  **Cálculo Centralizado**: `PhiEngramModel` calcula todos los hashes de todas las capas necesarias en un solo paso.
2.  **Distribución de Datos**: Los hashes se pasan como argumentos a las capas del decodificador, eliminando redundancia y minimizando transferencias entre CPU (donde ocurre el hash) y GPU (donde ocurre el cómputo).

---

## 4. Soporte para Generación Auto-regresiva (Incremental)

El mayor desafío técnico fue asegurar que `model.generate()` funcionara correctamente. Engram requiere n-gramas (historia de tokens). Implementamos:

1.  **Caché de Tokens (`engram_tokens`)**: Guardamos los últimos $N$ tokens en el objeto `past_key_values`. Cuando llega un token nuevo, lo concatenamos para formar los n-gramas de sufijo correctos.
2.  **Caché de Convolución (`engram_conv_cache`)**: Almacenamos los estados internos de la `ShortConv` por capa para que la convolución "recuerde" el pasado sin procesar toda la secuencia de nuevo.

---

## 5. Guía de Parámetros Técnicos

| Parámetro | Descripción | Valor Típico |
| :--- | :--- | :--- |
| `engram_layer_ids` | Capas donde se inyecta Engram. El paper sugiere capas tempranas (2, 15). | `[1, 15]` |
| `max_ngram_size` | Longitud máxima de n-grama (historia local). | `3` |
| `engram_vocab_size` | Tamaño de las tablas de hash. | `5 * vocab_size` |
| `n_embed_per_ngram` | Dimensión de los embeddings de memoria. | `512` |
| `hc_mult` | Multiplicador de hyper-conexión (para arquitecturas MoE). | `1` (en Phi-1) |

---

## 6. Referencias Matemáticas Principales

- **Fusión Residual**: $h^{(\ell)} \leftarrow h^{(\ell)} + Y + \text{Attn}(h^{(\ell)}) + \text{MLP}(h^{(\ell)})$
- **Activación Conv**: $Y = \text{SiLU}(\text{Conv1D}(\text{RMSNorm}(\tilde{V}))) + \tilde{V}$
- **Hashing**: $z_{t,n,k} = \varphi_{n,k}(g_{t,n})$ donde $\varphi$ es la función XOR-multiplicativa.

Esta implementación asegura que Phi-1 se beneficie de la "profundidad efectiva" adicional sin el coste computacional de añadir más capas de atención tradicionales.
