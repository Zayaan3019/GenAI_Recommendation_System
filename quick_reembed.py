# Create quick_reembed.py
import json
import numpy as np
from backend.app.config import get_settings
from backend.app.utils.embeddings_local import LocalEmbeddingGenerator

settings = get_settings()

# Load assessments
with open(f"{settings.DATA_DIR}/raw/assessments_final.json", 'r') as f:
    assessments = json.load(f)

print(f"Loaded {len(assessments)} assessments")

# Generate embeddings with correct model
emb_gen = LocalEmbeddingGenerator(model_name="all-mpnet-base-v2", cache_dir=f"{settings.DATA_DIR}/cache")

texts = []
for a in assessments:
    text = f"{a['assessment_name']} " * 5
    text += f"Type: {a['test_type']} "
    text += f"Description: {a['description']} "
    text += f"Skills: {' '.join(a['skills'])} "
    text += a['full_content'][:1500]
    texts.append(text)

print("Generating embeddings...")
embeddings_list = emb_gen.generate_batch_embeddings(texts, batch_size=16)
embeddings = np.array(embeddings_list, dtype=np.float32)

print(f"Generated: {embeddings.shape}")

# Save
data = {
    'assessments': assessments,
    'embeddings': embeddings.tolist(),
    'metadata': {
        'total': len(assessments),
        'dim': int(embeddings.shape[1]),
        'model': 'all-mpnet-base-v2'
    }
}

with open(settings.ASSESSMENTS_PATH, 'w') as f:
    json.dump(data, f, indent=2)

print(f"✓ Saved to {settings.ASSESSMENTS_PATH}")
