import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from evidently.report import Report
from evidently.metrics import EmbeddingsDriftMetric
from evidently.pipeline.column_mapping import ColumnMapping
import random
import os

# 1. Load a fast, lightweight embedding model
print("Loading sentence-transformer model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Prepare Reference Data (Real Contracts)
print("Loading reference data (CUAD test set)...")
with open("data/processed/test.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

# Extract just the raw text of the contracts, take 100 samples
reference_texts = [item["text"] for item in test_data[:100]]

# 3. Prepare Current Data (Simulate Drift)
print("Creating simulated production traffic (with drift)...")
# Take another 50 real contracts
current_texts = [item["text"] for item in test_data[100:150]]

# Simulate drift by adding 50 non-contract documents (news, recipes, casual text)
simulated_drift_texts = [
    "1 cup flour, 2 eggs, 1 cup milk. Mix well and fry on a pan to make pancakes.",
    "The local sports team won the championship last night in a thrilling 3-2 victory.",
    "I am writing to update my resume for the software engineer position.",
    "Patient presents with a mild fever and a cough. Prescribed rest and hydration.",
    "Breaking News: The stock market saw a massive surge today led by tech stocks.",
    "To reset your router, unplug it for 30 seconds and plug it back in.",
    "I really loved the movie! The acting was incredible and the plot twists were mind-blowing.",
    "List of items to buy: milk, bread, eggs, and toilet paper.",
    "Dear John, I hope this letter finds you well. It has been a long time since we last spoke.",
    "The quick brown fox jumps over the lazy dog."
] * 5  # Multiply to get 50 samples

current_texts.extend(simulated_drift_texts)
random.shuffle(current_texts)

# 4. Compute Embeddings
print("Computing embeddings for reference data...")
reference_embeddings = embedder.encode(reference_texts)

print("Computing embeddings for current data...")
current_embeddings = embedder.encode(current_texts)

# 5. Build Pandas DataFrames for Evidently
# Evidently requires the data in pandas dataframes. We'll store the embeddings as multiple columns.
ref_df = pd.DataFrame(reference_embeddings, columns=[f"emb_{i}" for i in range(reference_embeddings.shape[1])])
curr_df = pd.DataFrame(current_embeddings, columns=[f"emb_{i}" for i in range(current_embeddings.shape[1])])

# 6. Run Evidently Drift Report
print("Generating Evidently Drift Report...")
report = Report(metrics=[
    EmbeddingsDriftMetric("emb")  # "emb" prefix tells Evidently these columns belong to the same embedding
])

# Tell evidently which columns are the embeddings
column_mapping = ColumnMapping(
    embeddings={"emb": [f"emb_{i}" for i in range(reference_embeddings.shape[1])]}
)

report.run(reference_data=ref_df, current_data=curr_df, column_mapping=column_mapping)

report_path = "drift_report.html"
report.save_html(report_path)

print(f"\n✅ Drift report successfully generated at: {os.path.abspath(report_path)}")
print("Open this file in your browser to see the interactive drift dashboard!")
