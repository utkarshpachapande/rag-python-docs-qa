import time
import random

def generate_test_set(vector_db, n_samples, library_filter="all"):
    """
    Dynamically creates a test set by sampling real chunks from ChromaDB.
    This ensures the 'Test Samples' slider works for any count.
    """
    # Pull all documents and their associated library metadata
    all_data = vector_db.get()
    documents = all_data.get('documents', [])
    metadatas = all_data.get('metadatas', [])
    
    if not documents:
        return []

    indices = list(range(len(documents)))
    
    # Filter indices to match the user's selected library (e.g., 'pandas')
    if library_filter != "all":
        indices = [i for i in indices if metadatas[i].get('library') == library_filter]
    
    if not indices:
        return []

    # Randomly select a number of indices based on the 'Test Samples' slider
    selected_indices = random.sample(indices, min(n_samples, len(indices)))
    
    test_set = []
    for idx in selected_indices:
        lib = metadatas[idx].get('library', 'unknown')
        # Create a preview snippet for the question
        snippet = documents[idx][:80].replace("\n", " ")
        test_set.append({
            "question": f"Explain the following concept from {lib}: {snippet}...",
            "reference_context": documents[idx],
            "library": lib
        })
    return test_set

import pandas as pd
from sklearn.metrics import confusion_matrix
import plotly.express as px

def render_retrieval_confusion_matrix(metrics_log):
    """
    Creates a Confusion Matrix comparing requested library vs. retrieved library.
    """
    # 1. Extract data from the metrics log
    # Assumes your log stores 'expected_library' and 'retrieved_library'
    y_true = [entry.get('expected_library', 'Unknown') for entry in metrics_log]
    y_pred = [entry.get('retrieved_library', 'Unknown') for entry in metrics_log]
    
    # 2. Identify all unique labels for the matrix axes
    labels = sorted(list(set(y_true + y_pred)))
    
    # 3. Compute the matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # 4. Create Heatmap using Plotly
    fig = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Retrieved (Predicted)", y="Requested (Actual)", color="Count"),
        x=labels,
        y=labels,
        color_continuous_scale='Blues',
        aspect="auto"
    )
    
    fig.update_layout(
        title="📂 Retrieval Accuracy: Requested vs. Retrieved Library",
        paper_bgcolor='#111827',
        plot_bgcolor='#0f1e33',
        font=dict(color='#e2e8f0'),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

