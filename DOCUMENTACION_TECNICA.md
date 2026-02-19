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

---

## 7. Próximos Pasos Técnicos y Hoja de Ruta

Para llevar esta implementación de un prototipo verificado a un modelo productivo y entrenado, se deben seguir estos pasos exhaustivos:

### 7.1 Carga Progresiva de Pesos (Bootstrapping)
El modelo `PhiEngramForCausalLM` nace con el "backbone" (Phi-1) y la "memoria" (Engram). Para no perder el conocimiento previo de Microsoft:

1.  **Cargar Pesos del Backbone**: Utilizar `load_state_dict` con `strict=False` para cargar los pesos oficiales de Phi-1. Los módulos de Engram se ignorarán y permanecerán inicializados aleatoriamente.
2.  **Inicialización de Engram**:
    - Las tablas de embeddings deben inicializarse con una varianza pequeña (ej. `initializer_range=0.02`).
    - Las proyecciones de gating y valor deben ser inicializadas de forma que el gate $\alpha_t$ sea pequeño al principio, permitiendo que el backbone tome el control inicial mientras Engram aprende.

### 7.2 Estrategia de Entrenamiento (Sparsity-Aware Fine-tuning)
Engram no es un módulo "plug-and-play" que funcione sin entrenamiento. Requiere un proceso de fine-tuning específico:

-   **Fase 1: Warm-up de Memoria (Frozen Backbone)**:
    - Congelar todos los parámetros de Phi-1.
    - Entrenar únicamente las tablas de embeddings de Engram y sus capas de proyección (`key_projs`, `value_proj`).
    - **Objetivo**: Que el modelo aprenda a mapear n-gramas a conceptos útiles sin distorsionar las representaciones internas de Phi-1.
-   **Fase 2: Fine-tuning Conjunto (Unfrozen)**:
    - Descongelar todo el modelo.
    - Utilizar una tasa de aprendizaje (Learning Rate) para Engram que sea $5 \times$ superior a la del backbone.
    - Usar el optimizador **Adam** para los embeddings y **Muon** (si es posible) para las capas densas del backbone, siguiendo las recomendaciones de DeepSeek.

### 7.3 Optimización de Infraestructura: Prefetching y Offloading
En el paper original, se menciona que las tablas de Engram pueden ser masivas (superando la capacidad de la GPU).

-   **Implementación de Prefetching**: Dado que el hashing de n-gramas es determinista y solo depende de los tokens de entrada, se puede calcular el hash de la capa $L+n$ mientras la GPU aún procesa la capa $L$.
-   **Host Memory Offloading**: Mover las tablas de n-gramas a la RAM del sistema (CPU) y usar transferencias asíncronas vía PCIe para traer solo los vectores necesarios para el batch actual. Esto permite escalar el conocimiento del modelo a terabytes sin comprar más GPUs.

### 7.4 Evaluación de la "Profundidad Efectiva"
Para validar técnicamente la mejora, se deben realizar análisis de:
1.  **LogitLens**: Comparar la velocidad de convergencia de las predicciones entre Phi-1 puro y Phi-1 Engram.
2.  **CKA (Centered Kernel Alignment)**: Verificar si las capas tempranas de Phi-1 Engram se alinean representacionalmente con las capas tardías de Phi-1 estándar, confirmando que Engram está haciendo el "trabajo sucio" de reconstrucción de entidades.

### 7.5 Despliegue en el Hugging Face Hub
Para que el modelo sea cargable mediante `AutoModel.from_pretrained()`, se debe crear un archivo `configuration_phi_engram.py` y `modeling_phi_engram.py` dentro del repositorio del Hub, registrando las clases mediante `AutoConfig.register()` y `AutoModel.register()`.

---

## 8. Guía de Implementación del Entrenamiento

Hemos proporcionado dos scripts que materializan esta hoja de ruta:

### 8.1 Lógica de Carga (`load_phi_engram.py`)
Este script automatiza el **Weight Mapping**. Dado que la estructura de capas ha cambiado de `PhiDecoderLayer` a `PhiEngramDecoderLayer`, el mapeo de pesos estándar fallaría. Usamos:
- `AutoConfig` para replicar el espacio de parámetros de Phi-1.
- `PhiEngramForCausalLM` para instanciar la estructura aumentada.
- `model.load_state_dict(..., strict=False)` para transferir pesos del backbone ignorando los módulos de Engram ausentes en el original.

### 8.2 Lógica de Entrenamiento (`train_phi_engram.py`)
Implementa el algoritmo de **Warm-up de Memoria**:
1.  **Iteración de Gradientes**: Utiliza `named_modules()` para identificar y habilitar `requires_grad = True` solo en los sub-módulos que contienen la palabra "engram".
2.  **Optimización Diferencial**: Configura el optimizador para aplicar diferentes hiper-parámetros a distintos grupos de parámetros (Backbone vs Memoria), asegurando que el conocimiento estático se capture sin desestabilizar el razonamiento dinámico.

### 8.3 Interfaz Conversacional (`chat_phi_engram.py`)
Dado que Phi-1 es un modelo *CausalLM* base (no *Instruct*), la interfaz utiliza un **Chat Template** manual:
- Se concatena el prefijo `User:` y el sufijo `Assistant:` para inducir el comportamiento de respuesta.
- Se utiliza decodificación estocástica (`do_sample=True`) para permitir que la memoria Engram influya en la variabilidad de la salida.
