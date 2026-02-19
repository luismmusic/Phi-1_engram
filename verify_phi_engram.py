import torch
from phi1_engram import PhiEngramConfig, PhiEngramForCausalLM
from transformers import AutoTokenizer, DynamicCache

def verify():
    print("Iniciando verificación de Phi-1 con Engram...")

    # Configuración pequeña para prueba
    config = PhiEngramConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=8,
        engram_layer_ids=[1, 3],
        vocab_size=51200,
    )

    print("Instanciando modelo...")
    model = PhiEngramForCausalLM(config)
    model.eval()

    text = "Microsoft Phi-1 meets DeepSeek Engram."
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs.input_ids

    print(f"--- Prueba 1: Forward pass de secuencia completa ---")
    print(f"Tokens de entrada: {input_ids.shape}")
    with torch.no_grad():
        outputs_full = model(input_ids=input_ids)
    print(f"Logits shape: {outputs_full.logits.shape}")

    print(f"\n--- Prueba 2: Generación incremental (token a token) ---")
    past_key_values = DynamicCache()
    logits_incremental = []

    with torch.no_grad():
        for i in range(input_ids.shape[1]):
            curr_input_ids = input_ids[:, i:i+1]
            outputs = model(input_ids=curr_input_ids, past_key_values=past_key_values, use_cache=True)
            logits_incremental.append(outputs.logits)
            past_key_values = outputs.past_key_values

    logits_incremental = torch.cat(logits_incremental, dim=1)
    print(f"Logits incremental shape: {logits_incremental.shape}")

    # Verificar que los logits coincidan (con un pequeño margen de error por flotantes si los hay)
    diff = torch.abs(outputs_full.logits - logits_incremental).max()
    print(f"Diferencia máxima entre full e incremental: {diff.item()}")

    assert diff < 1e-4, f"Error: Los logits no coinciden. Dif: {diff.item()}"
    print("✅ Verificación completada con éxito. El soporte incremental funciona correctamente.")

if __name__ == "__main__":
    verify()
