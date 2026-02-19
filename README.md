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

## Ejecución en Google Colab

Para probar esta implementación en Google Colab, sigue estos pasos:

1. **Instalar dependencias**:
   Crea una celda y ejecuta:
   ```bash
   !pip install torch transformers tokenizers numpy sympy
   ```

2. **Cargar los archivos**:
   Puedes clonar este repositorio (si está en GitHub) o subir manualmente los archivos `phi1_engram.py` y `verify_phi_engram.py`.

   Si usas `git`:
   ```bash
   !git clone <URL_DE_TU_REPOSITORIO>
   %cd <NOMBRE_DEL_REPOSITORIO>
   ```

3. **Ejecutar Verificación**:
   Para confirmar que todo funciona correctamente en el entorno de Colab:
   ```bash
   !python verify_phi_engram.py
   ```

4. **Código de ejemplo rápido**:
   Puedes copiar este código directamente en una celda de Colab para una prueba inmediata:
   ```python
   import torch
   from phi1_engram import PhiEngramConfig, PhiEngramForCausalLM
   from transformers import AutoTokenizer

   # Configuración de prueba
   config = PhiEngramConfig(
       hidden_size=256,
       num_hidden_layers=2,
       vocab_size=51200
   )
   model = PhiEngramForCausalLM(config)

   tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1")
   inputs = tokenizer("Hello Engram!", return_tensors="pt")

   with torch.no_grad():
       outputs = model(**inputs)
   print("Logits shape:", outputs.logits.shape)
   ```

## Ejecución en Kaggle

En Kaggle, el proceso es muy similar al de Colab, con algunas configuraciones adicionales:

1. **Configuración del Kernel**:
   - En el panel derecho (**Settings**), asegúrate de que **Internet on** esté activado.
   - Selecciona un **Accelerator**: GPU T4 x2 o GPU P100 son recomendados.

2. **Instalar dependencias**:
   ```bash
   !pip install torch transformers tokenizers numpy sympy
   ```

3. **Uso de GPU**:
   Asegúrate de mover el modelo y los datos a la GPU para un rendimiento óptimo:
   ```python
   device = "cuda" if torch.cuda.is_available() else "cpu"
   model.to(device)
   inputs = {k: v.to(device) for k, v in inputs.items()}
   ```

## Integración con Hugging Face Hub

Puedes subir este modelo al Hub para que otros lo usen fácilmente:

1. **Subir archivos**:
   Asegúrate de subir `phi1_engram.py` junto con los archivos de pesos (`pytorch_model.bin` o `model.safetensors`).

2. **Configuración para `trust_remote_code`**:
   Para que otros puedan cargar tu modelo sin instalar localmente el archivo `.py`, asegúrate de registrar las clases en tu script o usar la opción `trust_remote_code=True`.

3. **Ejemplo de carga desde el Hub**:
   ```python
   from transformers import AutoModelForCausalLM, AutoConfig
   from phi1_engram import PhiEngramConfig, PhiEngramForCausalLM

   # Suponiendo que el modelo está en 'tu-usuario/phi1-engram'
   model = AutoModelForCausalLM.from_pretrained(
       "tu-usuario/phi1-engram",
       trust_remote_code=True
   )
   ```
   *Nota: Para que el Hub reconozca automáticamente las clases, debes incluir las referencias adecuadas en el archivo `config.json` del repositorio.*