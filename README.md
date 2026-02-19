# Phi-1 Engram

Esta es una implementación de Microsoft Phi-1 que incorpora el módulo **Engram** de DeepSeek.

## Características
- Integración del módulo Engram para memoria de n-gramas condicional.
- Arquitectura compatible con los pesos de Microsoft Phi-1.
- Implementación completa de `CompressedTokenizer`, `NgramHashMapping` y el mecanismo de gating.

## Uso
Para instanciar el modelo:
```python
from phi1_engram import PhiEngramConfig, PhiEngramForCausalLM

config = PhiEngramConfig(
    engram_layer_ids=[1, 15], # Capas donde se inserta Engram
    # ... otros parámetros de Phi-1
)
model = PhiEngramForCausalLM(config)
```

## Verificación
Puedes ejecutar el script de verificación para confirmar que el modelo funciona correctamente:
```bash
python verify_phi_engram.py
```

## Orígenes y Referencias
Esta implementación integra:
- **Microsoft Phi-1**: Basado en la arquitectura oficial soportada por la biblioteca `transformers`.
- **DeepSeek Engram**: Lógica adaptada del paper "Conditional Memory via Scalable Lookup" (2026).

*Nota: No se clonaron los repositorios externos para asegurar una implementación optimizada, portátil y compatible con la generación incremental paso a paso.*

## Próximos Pasos (Next Steps)

Para aprovechar al máximo esta arquitectura, se recomiendan los siguientes pasos:

### 1. Carga de Pesos Pre-entrenados
El backbone de Phi-1 puede cargarse con pesos oficiales de Microsoft, mientras que los módulos de Engram se inicializarán aleatoriamente.
```python
from phi1_engram import PhiEngramForCausalLM, PhiEngramConfig
from transformers import AutoConfig

# Cargar configuración base
base_config = AutoConfig.from_pretrained("microsoft/phi-1")
# Combinar con parámetros de Engram
config = PhiEngramConfig(**base_config.to_dict(), engram_layer_ids=[1, 15])
# Instanciar modelo
model = PhiEngramForCausalLM(config)

# Opcional: Cargar pesos del backbone desde el modelo original
# state_dict = torch.load("path_to_phi1_weights")
# model.load_state_dict(state_dict, strict=False)
```

### 2. Entrenamiento / Fine-tuning
Dado que los módulos de Engram son nuevos, requieren entrenamiento.
- **Estrategia A (Warm-up)**: Congelar el backbone y entrenar solo las tablas de embeddings de Engram y sus proyecciones.
- **Estrategia B (Full Fine-tuning)**: Entrenar todo el modelo en un corpus de conocimiento denso (ej. Wikipedia o libros) para que Engram aprenda los patrones de n-gramas.

### 3. Optimización de Memoria (Offloading)
Si el vocabulario de Engram crece demasiado, implementa el "prefetching" mencionado en el paper de DeepSeek para mover las tablas de embeddings a la RAM de la CPU, liberando VRAM en la GPU.

### 4. Evaluación Comparativa
Evaluar en benchmarks de razonamiento (BBH) y matemáticas (MATH) para verificar la ganancia de "profundidad efectiva" descrita en el whitepaper técnico.