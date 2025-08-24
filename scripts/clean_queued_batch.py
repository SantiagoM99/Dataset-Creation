import requests
from dotenv import load_dotenv

# Reemplaza con tu API key
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = "https://api.openai.com/v1"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def list_batches():
    """Lista todos los batch jobs de la cuenta."""
    url = f"{BASE_URL}/batches"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

def cancel_batch(batch_id):
    """Cancela un batch job por ID."""
    url = f"{BASE_URL}/batches/{batch_id}/cancel"
    response = requests.post(url, headers=headers)
    if response.status_code == 200:
        print(f"✅ Batch {batch_id} cancelado.")
    else:
        print(f"⚠️ Error al cancelar {batch_id}: {response.text}")

def main():
    data = list_batches()
    batches = data.get("data", [])
    
    print(f"Se encontraron {len(batches)} batch jobs.")

    for batch in batches:
        batch_id = batch["id"]
        status = batch.get("status", "desconocido")
        print(f"- {batch_id}: {status}")

        # Cancelar si sigue en progreso o validando
        if status in ["in_progress", "validating"]:
            cancel_batch(batch_id)

if __name__ == "__main__":
    main()
