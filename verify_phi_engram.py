import torch
from phi1_engram import PhiEngramConfig, PhiEngramForCausalLM
from transformers import AutoTokenizer, DynamicCache

def verify():
    print("--------------------------------------------------")
    print("INICIANDO VERIFICACIÓN DE PHI-1 CON ENGRAM")
    print("--------------------------------------------------")

    # 1. Configuración del modelo (versión pequeña para que corra rápido)
    print("[1/5] Configurando el modelo de prueba...")
    config = PhiEngramConfig(
        hidden_size=256,        # Tamaño de las capas
        intermediate_size=512,  # Tamaño intermedio
        num_hidden_layers=4,    # Solo 4 capas para la prueba
        num_attention_heads=8,
        num_key_value_heads=8,
        engram_layer_ids=[1, 3], # Pondremos Engram en las capas 1 y 3
        vocab_size=51200,       # Tamaño del diccionario estándar
    )

    # 2. Instanciación
    print("[2/5] Creando el modelo en memoria...")
    # Usamos half precision si hay GPU para ser más eficientes
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = PhiEngramForCausalLM(config).to(device=device, dtype=dtype)
    model.eval() # Ponemos el modelo en modo lectura (evaluación)

    # 3. Preparación del texto
    print("[3/5] Preparando el texto de prueba...")
    text = "Microsoft Phi-1 meets DeepSeek Engram."
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs.input_ids

    # --- PRUEBA 1 ---
    print("\n--- PRUEBA 1: Procesamiento por Bloques (Batch) ---")
    print(f"Enviando {input_ids.shape[1]} tokens a la vez...")
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs_full = model(input_ids=input_ids)
    print(f"¡Hecho! Resultado obtenido: {outputs_full.logits.shape}")

    # --- PRUEBA 2 ---
    print("\n--- PRUEBA 2: Generación Paso a Paso (Incremental) ---")
    print("Esto simula cómo el modelo escribe una palabra a la vez...")
    past_key_values = DynamicCache() # Aquí se guarda la memoria de lo que ya se leyó
    logits_incremental = []

    with torch.no_grad():
        for i in range(input_ids.shape[1]):
            curr_input_ids = input_ids[:, i:i+1].to(device) # Tomamos solo el token actual
            # El modelo usa el token actual + lo que recuerda (past_key_values)
            outputs = model(input_ids=curr_input_ids, past_key_values=past_key_values, use_cache=True)
            logits_incremental.append(outputs.logits)
            past_key_values = outputs.past_key_values # Actualizamos lo que el modelo recuerda

    logits_incremental = torch.cat(logits_incremental, dim=1)
    print(f"¡Hecho! Generación incremental completada.")

    # 4. Comparación de resultados
    print("\n[4/5] Comparando precisión matemática...")
    # Calculamos la diferencia entre los dos métodos
    diff = torch.abs(outputs_full.logits - logits_incremental).max()
    print(f"Diferencia máxima detectada: {diff.item():.2e}")

    # 5. Resultado final
    print("\n[5/5] RESULTADO FINAL:")
    if diff < 1e-4:
        print("✅ ¡ÉXITO TOTAL!")
        print("El modelo es matemáticamente consistente.")
        print("Puedes usarlo para generar texto con total confianza.")
    else:
        print("❌ ERROR: Los resultados no coinciden.")
        print("Revisa la implementación del caché de Engram.")
    print("--------------------------------------------------\n")

if __name__ == "__main__":
    verify()
