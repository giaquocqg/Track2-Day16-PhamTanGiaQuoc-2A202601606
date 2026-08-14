import time
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import lightgbm as lgb

def run_benchmark():
    print("==========================================================")
    print("  LAB 16: Credit Card Fraud Detection - LightGBM Benchmark")
    print("==========================================================")
    
    # 1. Load Data
    t0_load = time.time()
    csv_path = "creditcard.csv"
    print(f"[*] Loading dataset from '{csv_path}'...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        df = pd.read_csv("ml-benchmark/creditcard.csv")
    load_time_sec = time.time() - t0_load
    dataset_rows = int(df.shape[0])
    print(f"[+] Data loaded in {load_time_sec:.4f}s. Shape: {df.shape}")
    
    # 2. Train/Test Split (Stratified Sampling)
    X = df.drop(columns=["Class"])
    y = df["Class"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[*] Train set: {X_train.shape[0]} rows, Test set: {X_test.shape[0]} rows")
    print(f"[*] Class imbalance: {y.value_counts().to_dict()} (Fraud ratio: {y.mean()*100:.3f}%)")

    # 3. Model Training
    print("[*] Training LightGBM Classifier...")
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    t0_train = time.time()
    model.fit(X_train, y_train)
    train_time_sec = time.time() - t0_train
    print(f"[+] Model trained in {train_time_sec:.4f}s")
    
    # 4. Evaluation
    print("[*] Evaluating on test set...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    auc_roc = float(roc_auc_score(y_test, y_pred_proba))
    accuracy = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred))
    recall = float(recall_score(y_test, y_pred))
    
    print("\n--- Classification Metrics ---")
    print(f"AUC-ROC   : {auc_roc:.6f}")
    print(f"Accuracy  : {accuracy:.6f}")
    print(f"Precision : {precision:.6f}")
    print(f"Recall    : {recall:.6f}")
    print(f"F1-Score  : {f1:.6f}")
    
    # 5. Latency & Throughput Benchmark
    print("\n[*] Measuring inference latency and throughput...")
    sample_single = X_test.iloc[[0]]
    sample_1000 = X_test.iloc[:1000]
    
    # Warmup
    for _ in range(10):
        _ = model.predict_proba(sample_single)
        
    # Measure 1-row latency (average over 100 runs)
    latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = model.predict_proba(sample_single)
        latencies.append((time.perf_counter() - t0) * 1000) # in ms
    avg_latency_1_row_ms = float(np.mean(latencies))
    
    # Measure 1000-row batch throughput
    t0_batch = time.perf_counter()
    _ = model.predict_proba(sample_1000)
    batch_time_sec = time.perf_counter() - t0_batch
    throughput_samples_per_sec = float(1000.0 / batch_time_sec)
    
    print(f"Inference Latency (1 row)   : {avg_latency_1_row_ms:.4f} ms")
    print(f"Throughput                  : {throughput_samples_per_sec:.2f} rows/sec")

    # 6. Save results matching exact Lab schema
    results = {
        "cloud": "gcp",
        "instance_type": "e2-medium",
        "dataset_rows": dataset_rows,
        "load_time_seconds": round(load_time_sec, 4),
        "training_time_seconds": round(train_time_sec, 4),
        "auc_roc": round(auc_roc, 6),
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1_score": round(f1, 6),
        "inference_latency_ms_one_row": round(avg_latency_1_row_ms, 4),
        "inference_throughput_rows_per_second": round(throughput_samples_per_sec, 2)
    }
    
    with open("benchmark_result.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n[+] Benchmark results saved to 'benchmark_result.json'")
    print("==========================================================")

if __name__ == "__main__":
    run_benchmark()
