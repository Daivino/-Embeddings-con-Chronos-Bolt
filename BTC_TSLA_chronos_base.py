import pandas as pd
import torch
from chronos import BaseChronosPipeline

# --- CONFIGURACIÓN PERSONALIZADA ---
FILES = {
    'BTC': "hf://datasets/TheFinAI/CLEF_Task3_Trading/data/BTC-00000-of-00001.parquet",
    'TSLA': "hf://datasets/TheFinAI/CLEF_Task3_Trading/data/TSLA-00000-of-00001.parquet"
}

# DICCIONARIO DE VENTANAS:
# Aquí defines cuántos días quieres para cada uno específicamente
ASSET_WINDOWS = {
    'TSLA': 5,  # 1 Semana bursátil
    'BTC': 8    # 1 Semana crypto + confirmación
}

MODEL_NAME = "amazon/chronos-bolt-base" 

def get_custom_embeddings():
    print(f" Iniciando extracción personalizada...")
    
    # Cargar modelo
    pipeline = BaseChronosPipeline.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    bolt_model = pipeline.model

    for asset, url in FILES.items():
        # Obtenemos la ventana específica para este activo
        # Si no está en el diccionario, usamos 30 por defecto
        window_size = ASSET_WINDOWS.get(asset, 30)
        
        print(f"\n{'='*40}")
        print(f"Procesando: {asset}")
        print(f" Ventana seleccionada: Últimos {window_size} días")
        
        try:
            # Cargar y ordenar
            df = pd.read_parquet(url)
            
            # Asegurar orden cronológico (Vital para que 'últimos días' sea real)
            if 'date' in df.columns: # Asumiendo que hay columna fecha
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
            
            full_prices = df['prices'].values.astype(float)
            
            # --- CORTE DE DATOS (SLICING) ---
            if len(full_prices) < window_size:
                print(f"  Error: No hay suficientes datos ({len(full_prices)}) para ventana de {window_size}.")
                continue
                
            # Tomamos los últimos N días definidos en ASSET_WINDOWS
            window_prices = full_prices[-window_size:]
            
            # Tensor
            context_tensor = torch.tensor(window_prices).unsqueeze(0).to(
                device=bolt_model.device, 
                dtype=bolt_model.dtype
            )
            
            # Extracción
            with torch.no_grad():
                raw_output = bolt_model.encode(context_tensor)
                embeddings = raw_output[0]
            
            # Guardar
            emb_numpy = embeddings.squeeze(0).float().cpu().numpy()
            
            # Nombre de archivo descriptivo
            output_filename = f"{asset}_embeddings_{window_size}d.parquet"
            pd.DataFrame({'embedding': list(emb_numpy)}).to_parquet(output_filename)
            
            print(f"   Guardado: {output_filename}")
            print(f"   Shape: {emb_numpy.shape}")
            print(f"   Logica: {'Semana Bursátil' if asset=='TSLA' else 'Ciclo Crypto'}")

        except Exception as e:
            print(f" Error en {asset}: {str(e)}")

if __name__ == "__main__":
    get_custom_embeddings()