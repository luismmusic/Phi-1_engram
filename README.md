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