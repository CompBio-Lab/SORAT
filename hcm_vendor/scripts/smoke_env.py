#!/usr/bin/env python3
"""Environment smoke test: imports, a tiny XGBoost + torch MLP fit, and TreeSHAP."""

import importlib
import json
import platform
import sys
import time

PACKAGES = [
    "numpy", "pandas", "pyarrow", "scipy", "sklearn", "xgboost", "shap", "torch",
    "matplotlib", "seaborn", "yaml", "joblib", "pytest", "statsmodels", "SimpleITK",
]


def main() -> None:
    versions = {"python": sys.version.split()[0], "host": platform.node()}
    for name in PACKAGES:
        start = time.time()
        module = importlib.import_module(name)
        versions[name] = getattr(module, "__version__", "?")
        print(f"import {name} {versions[name]} ({time.time() - start:.1f}s)", flush=True)

    import numpy as np
    import shap
    import torch
    import xgboost
    from sklearn.datasets import make_classification
    from sklearn.metrics import roc_auc_score

    X, y = make_classification(n_samples=120, n_features=20, random_state=0)

    booster = xgboost.XGBClassifier(n_estimators=50, max_depth=2, tree_method="hist", n_jobs=1)
    booster.fit(X, y)
    xgb_auc = roc_auc_score(y, booster.predict_proba(X)[:, 1])
    shap_values = shap.TreeExplainer(booster).shap_values(X[:10])

    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(20, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    for _ in range(200):
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(xt), yt)
        loss.backward()
        optimizer.step()
    mlp_auc = roc_auc_score(y, torch.sigmoid(model(xt)).detach().numpy().ravel())

    checks = {
        "xgb_train_auc": round(float(xgb_auc), 3),
        "mlp_train_auc": round(float(mlp_auc), 3),
        "shap_shape": list(np.asarray(shap_values).shape),
        "torch_cuda_available": bool(torch.cuda.is_available()),
    }
    print(json.dumps({"versions": versions, "checks": checks}, indent=2))
    assert checks["xgb_train_auc"] > 0.9 and checks["mlp_train_auc"] > 0.9
    assert checks["shap_shape"] == [10, 20]
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
