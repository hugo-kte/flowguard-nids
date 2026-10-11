"""Display names shared by the report figures and the dashboard."""

MODEL_LABELS = {
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "Gradient Boosting",
    "isolation_forest": "Isolation Forest",
    "autoencoder": "Autoencoder",
    "hybrid": "Hybrid (GB + AE)",
}


def label(name: str) -> str:
    return MODEL_LABELS.get(name, name.replace("_", " ").title())
