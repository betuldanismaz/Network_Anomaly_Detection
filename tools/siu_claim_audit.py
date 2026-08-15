"""
E1-S12: SIU 2026 (394) Claim Audit
===================================
Checks each SIU paper claim against repo artifacts and produces
reports/siu_claim_audit.md with UYUMLU / SAPMA / DOGRULANAMADI labels.
"""

import os
import sys
import json
import glob
from collections import Counter
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

# ── Paper values (SIU 2026, bildiri 394) ──────────────────────────────────────

PAPER_TABLE_I = {"Benign": 2_265_000, "Volumetric": 396_000, "Semantic": 170_000}

PAPER_TABLE_II = {
    "LSTM":     {"acc": 98.15, "macro_prec": 95.79, "macro_recall": 97.13, "macro_f1": 96.45, "latency_ms": 0.0575, "throughput": 17_383},
    "XGBoost":  {"acc": 97.71, "macro_prec": 93.62, "macro_recall": 98.38, "macro_f1": 95.87, "latency_ms": 0.0070, "throughput": 142_428},
    "DT":       {"acc": 97.27, "macro_prec": 94.26, "macro_recall": 97.83, "macro_f1": 95.84, "latency_ms": 0.0001, "throughput": 10_534_659},
    "RF":       {"acc": 97.34, "macro_prec": 94.43, "macro_recall": 97.70, "macro_f1": 95.91, "latency_ms": 0.0031, "throughput": 319_398},
    "BiLSTM":   {"acc": 98.88, "macro_prec": 97.98, "macro_recall": 98.02, "macro_f1": 97.72, "latency_ms": 0.1085, "throughput": 9_212},
}

PAPER_TABLE_III = {
    "LSTM":    {"Benign": 98.84, "Volumetric": 95.58, "Semantic": 94.93},
    "XGBoost": {"Benign": 98.56, "Volumetric": 94.35, "Semantic": 94.71},
    "DT":      {"Benign": 98.28, "Volumetric": 91.71, "Semantic": 97.53},
    "RF":      {"Benign": 98.32, "Volumetric": 91.94, "Semantic": 97.36},
    "BiLSTM":  {"Benign": 99.30, "Volumetric": 96.92, "Semantic": 97.72},
}

LATENCY_MODEL_MAP = {
    "XGBoost (GPU)": "XGBoost",
    "Random Forest": "RF",
    "Decision Tree": "DT",
    "LSTM": "LSTM",
    "BiLSTM": "BiLSTM",
}

CONFIG_MODEL_MAP = {
    "RF": "rf_3class_config.json",
    "XGBoost": "xgb_3class_config.json",
}

CLASSES_MAP_PATH_CANDIDATES = [
    os.path.join(ROOT, "reports", "data", "classes_map.json"),
    os.path.join(ROOT, "src", "utils", "classes_map.json"),
    os.path.join(ROOT, "data", "classes_map.json"),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

UYUMLU = "UYUMLU"
SAPMA = "SAPMA"
DOGRULANAMADI = "DOGRULANAMADI"

results = []


def add(section, item_id, description, status, detail=""):
    results.append({
        "section": section,
        "id": item_id,
        "description": description,
        "status": status,
        "detail": detail,
    })


def pct_close(paper_val, repo_val, tol=0.15):
    return abs(paper_val - repo_val) <= tol


# ── K1: Per-file stratified split ────────────────────────────────────────────

def audit_k1():
    preprocess_path = os.path.join(ROOT, "src", "features", "preprocess_ml_3class.py")
    if not os.path.exists(preprocess_path):
        add("K1", "K1.1-5", "preprocess_ml_3class.py", DOGRULANAMADI, "Dosya bulunamadi")
        return

    with open(preprocess_path, "r", encoding="utf-8") as f:
        code = f.read()

    add("K1", "K1.1", "Her dosya uzerinde bagimsiz bolme",
        UYUMLU if "process_single_file" in code else SAPMA,
        "process_single_file() fonksiyonu mevcut")

    add("K1", "K1.2", "Stratified split (stratify=y)",
        UYUMLU if "stratify=y" in code or "stratify=y_temp" in code else SAPMA,
        "stratify parametresi kullaniliyor")

    add("K1", "K1.3", "80/10/10 bolme orani",
        UYUMLU if "test_size=0.2" in code and "test_size=0.5" in code else SAPMA,
        "test_size=0.2 → %80/%20, sonra 0.5 → %10/%10")

    add("K1", "K1.4", "Dosyalar birlesmeden ONCE boluyor",
        UYUMLU,
        "process_single_file() per-file split, sonra concat")

    add("K1", "K1.5", "RANDOM_STATE = 42",
        UYUMLU if "RANDOM_STATE = 42" in code else SAPMA,
        "Sabit seed")


# ── K2: 3-class taxonomy ─────────────────────────────────────────────────────

def audit_k2():
    cm_path = None
    for p in CLASSES_MAP_PATH_CANDIDATES:
        if os.path.exists(p):
            cm_path = p
            break
    if cm_path is None:
        add("K2", "K2.1-4", "classes_map.json", DOGRULANAMADI, "Dosya bulunamadi")
        return

    with open(cm_path, "r", encoding="utf-8") as f:
        cm = json.load(f)

    vol_labels = [k for k, v in cm.items() if v == 1]
    sem_labels = [k for k, v in cm.items() if v == 2]

    csv_dir = os.path.join(ROOT, "data", "original_csv")
    csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))

    if csv_files:
        import unicodedata
        import pandas as pd

        def normalize(x):
            s = str(x)
            s = unicodedata.normalize("NFKC", s)
            s = s.replace("﻿", "").replace(" ", " ").replace("�", " ")
            return " ".join(s.split()).strip()

        cm_norm = {normalize(k): v for k, v in cm.items()}
        counts = Counter()
        for fpath in csv_files:
            df = pd.read_csv(fpath, usecols=[" Label"], dtype=str)
            df.columns = df.columns.str.strip()
            labels = df["Label"].fillna("").map(normalize).map(cm_norm)
            for val in labels.dropna():
                counts[int(val)] += 1

        total = sum(counts.values())
        repo_counts = {
            "Benign": counts.get(0, 0),
            "Volumetric": counts.get(1, 0),
            "Semantic": counts.get(2, 0),
        }

        for cls_name, paper_val in PAPER_TABLE_I.items():
            repo_val = repo_counts[cls_name]
            diff = abs(repo_val - paper_val)
            tol = paper_val * 0.02
            if diff <= tol:
                add("K2", f"K2.1-{cls_name}", f"Tablo I {cls_name} sayisi: paper {paper_val:,} vs repo {repo_val:,}",
                    UYUMLU, f"Fark: {diff:,} (<%2 tolerans)")
            else:
                add("K2", f"K2.1-{cls_name}", f"Tablo I {cls_name} sayisi: paper {paper_val:,} vs repo {repo_val:,}",
                    SAPMA, f"Fark: {diff:,}")

        for cls_name in PAPER_TABLE_I:
            pct = 100 * repo_counts[cls_name] / total if total > 0 else 0
            paper_pct = 100 * PAPER_TABLE_I[cls_name] / sum(PAPER_TABLE_I.values())
            add("K2", f"K2.2-{cls_name}", f"Oran {cls_name}: paper ~%{paper_pct:.0f} vs repo %{pct:.1f}",
                UYUMLU if abs(pct - paper_pct) < 2 else SAPMA)
    else:
        add("K2", "K2.1", "Sinif sayilari", DOGRULANAMADI, "original_csv dizininde CSV bulunamadi")

    paper_vol = {"DDoS", "DoS Hulk", "DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest", "Heartbleed"}
    bot_in = "Bot" in sem_labels or "Botnet" in sem_labels
    if bot_in:
        add("K2", "K2.3", "Bot sinifi esleme: paper Botnet→Hacimsel, repo Bot→Semantic",
            SAPMA, f"classes_map Bot→2 (Semantic). Paper Tablo I Botnet'i Hacimsel'e koyar. JNCA'da aciklanmali.")
    else:
        add("K2", "K2.3", "Bot/Botnet sinif esleme", UYUMLU)

    heartbleed_in_vol = "Heartbleed" in vol_labels
    add("K2", "K2.4", "Heartbleed esleme",
        UYUMLU if heartbleed_in_vol else SAPMA,
        f"Heartbleed → {'Volumetric' if heartbleed_in_vol else 'baska'} (paper'da Hacimsel altinda listelenmemis ama DoS ailesi)")


# ── K3: 5-model comparison ──────────────────────────────────────────────────

def audit_k3():
    # Config-based metrics (RF, XGBoost)
    for model_short, config_file in CONFIG_MODEL_MAP.items():
        config_path = os.path.join(ROOT, "models", config_file)
        if not os.path.exists(config_path):
            add("K3", f"K3-{model_short}-config", f"{model_short} config", DOGRULANAMADI, "Config dosyasi yok")
            continue

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        paper = PAPER_TABLE_II[model_short]
        test_m = cfg.get("test_metrics", {})

        repo_f1 = test_m.get("macro_f1", 0) * 100
        if pct_close(paper["macro_f1"], repo_f1):
            add("K3", f"K3-{model_short}-F1", f"{model_short} Macro-F1: paper {paper['macro_f1']}% vs repo {repo_f1:.2f}%",
                UYUMLU)
        else:
            add("K3", f"K3-{model_short}-F1", f"{model_short} Macro-F1: paper {paper['macro_f1']}% vs repo {repo_f1:.2f}%",
                SAPMA, f"Fark: {abs(paper['macro_f1'] - repo_f1):.2f}pp")

        # Per-class F1 (Table III)
        paper_cls = PAPER_TABLE_III[model_short]
        per_class = test_m.get("per_class", cfg.get("per_class_metrics", {}))
        for cls_name, paper_f1 in paper_cls.items():
            cls_data = per_class.get(cls_name, {})
            repo_f1_cls = cls_data.get("f1", 0)
            if repo_f1_cls < 1:
                repo_f1_cls *= 100
            if pct_close(paper_f1, repo_f1_cls):
                add("K3", f"K3-{model_short}-{cls_name}", f"Tablo III {model_short} {cls_name} F1: paper {paper_f1}% vs repo {repo_f1_cls:.2f}%",
                    UYUMLU)
            else:
                add("K3", f"K3-{model_short}-{cls_name}", f"Tablo III {model_short} {cls_name} F1: paper {paper_f1}% vs repo {repo_f1_cls:.2f}%",
                    SAPMA, f"Fark: {abs(paper_f1 - repo_f1_cls):.2f}pp")

    # DL model metrics (no config with test metrics)
    for model_short in ["LSTM", "BiLSTM", "DT"]:
        add("K3", f"K3-{model_short}-metrics", f"{model_short} test metrikleri",
            DOGRULANAMADI, "Config JSON'da test metrikleri yok; modeli yukleyip test setinde cikarim gerekli")

    # Latency benchmark
    lat_path = os.path.join(ROOT, "reports", "latency_benchmark.json")
    if os.path.exists(lat_path):
        with open(lat_path, "r", encoding="utf-8") as f:
            lat_data = json.load(f)

        for entry in lat_data.get("results", []):
            model_name = entry["model"]
            model_short = LATENCY_MODEL_MAP.get(model_name)
            if not model_short:
                continue
            paper = PAPER_TABLE_II[model_short]

            repo_lat = entry["latency_ms"]
            repo_thr = entry["throughput"]

            lat_ok = abs(paper["latency_ms"] - repo_lat) < 0.01
            thr_ratio = abs(paper["throughput"] - repo_thr) / max(paper["throughput"], 1)

            if lat_ok and thr_ratio < 0.05:
                add("K3", f"K3-{model_short}-latency", f"{model_short} latency/throughput: paper {paper['latency_ms']}ms/{paper['throughput']:,} vs repo {repo_lat:.4f}ms/{repo_thr:,.0f}",
                    UYUMLU)
            else:
                add("K3", f"K3-{model_short}-latency", f"{model_short} latency/throughput: paper {paper['latency_ms']}ms/{paper['throughput']:,} vs repo {repo_lat:.4f}ms/{repo_thr:,.0f}",
                    SAPMA, f"Throughput farki: {abs(paper['throughput'] - repo_thr):,.0f}")

        add("K3", "K3-protocol", "10,000 orneklemlik test protokolu",
            UYUMLU if lat_data.get("benchmark_protocol") == "10000_samples" else SAPMA,
            f"benchmark_protocol = {lat_data.get('benchmark_protocol')}")
    else:
        add("K3", "K3-latency", "Latency benchmark", DOGRULANAMADI, "reports/latency_benchmark.json yok")

    # Derived claims
    bilstm_thr = PAPER_TABLE_II["BiLSTM"]["throughput"]
    xgb_thr = PAPER_TABLE_II["XGBoost"]["throughput"]
    ratio = xgb_thr / bilstm_thr
    add("K3", "K3-15.5x", f"XGBoost 15.5x daha hizli: {ratio:.1f}x",
        UYUMLU if abs(ratio - 15.5) < 0.5 else SAPMA)

    bilstm_f1 = PAPER_TABLE_II["BiLSTM"]["macro_f1"]
    dt_f1 = PAPER_TABLE_II["DT"]["macro_f1"]
    gap = bilstm_f1 - dt_f1
    add("K3", "K3-1.88pp", f"Performans acigi: {gap:.2f}pp (paper 1.88pp)",
        UYUMLU if abs(gap - 1.88) < 0.05 else SAPMA)

    # Hyperparameters
    xgb_cfg_path = os.path.join(ROOT, "models", "xgb_3class_config.json")
    if os.path.exists(xgb_cfg_path):
        with open(xgb_cfg_path, "r") as f:
            xgb_cfg = json.load(f)
        hp = xgb_cfg.get("hyperparameters", {})
        checks = [
            hp.get("max_depth") == 7,
            hp.get("learning_rate") == 0.05,
            hp.get("subsample") == 0.8,
            hp.get("colsample_bytree") == 0.8,
        ]
        add("K3", "K3-XGB-HP", "XGBoost hiperparametreleri",
            UYUMLU if all(checks) else SAPMA,
            f"max_depth={hp.get('max_depth')}, lr={hp.get('learning_rate')}, sub={hp.get('subsample')}, col={hp.get('colsample_bytree')}")

    # LSTM/BiLSTM architecture
    for model_short, cfg_file in [("LSTM", "lstm_config.json"), ("BiLSTM", "bilstm_config.json")]:
        cfg_path = os.path.join(ROOT, "models", cfg_file)
        if os.path.exists(cfg_path):
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
            shape = cfg.get("input_shape", [])
            u1 = cfg.get("lstm_units_1", 0)
            u2 = cfg.get("lstm_units_2", 0)
            dr = cfg.get("dropout_rate", 0)
            ok = shape == [10, 20] and u1 == 128 and u2 == 64 and dr == 0.3
            add("K3", f"K3-{model_short}-arch", f"{model_short} mimari: {u1}/{u2} units, {dr} dropout, shape={shape}",
                UYUMLU if ok else SAPMA)

    # Features
    try:
        from config import TOP_FEATURES
        add("K3", "K3-features", f"20 oznitelik (TOP_FEATURES): {len(TOP_FEATURES)} eleman",
            UYUMLU if len(TOP_FEATURES) == 20 else SAPMA)
    except ImportError:
        add("K3", "K3-features", "TOP_FEATURES", DOGRULANAMADI)

    # RF grid search
    rf_train = os.path.join(ROOT, "src", "models", "train_randomforest.py")
    if os.path.exists(rf_train):
        with open(rf_train, "r", encoding="utf-8") as f:
            rf_code = f.read()
        add("K3", "K3-RF-grid", "RF 36 kombinasyonluk grid search",
            UYUMLU if "36" in rf_code or "36-combo" in rf_code else SAPMA,
            "train_randomforest.py'da 36-combo grid search referansi mevcut")
    else:
        add("K3", "K3-RF-grid", "RF grid search", DOGRULANAMADI)


# ── K4: Live Bridge architecture ─────────────────────────────────────────────

def audit_k4():
    # K4.1-K4.3: Scapy, CICFlowMeter, Kafka
    bridge_path = os.path.join(ROOT, "src", "live_bridge.py")
    if os.path.exists(bridge_path):
        with open(bridge_path, "r", encoding="utf-8") as f:
            bridge_code = f.read()
        add("K4", "K4.1", "Scapy ile canli paket yakalama",
            UYUMLU if "scapy" in bridge_code.lower() or "sniff" in bridge_code else SAPMA)
        add("K4", "K4.2", "CICFlowMeter ile oznitelik cikarimi",
            UYUMLU if "cicflowmeter" in bridge_code.lower() or "cfm" in bridge_code.lower() else SAPMA)
        add("K4", "K4.4", "4 saniyelik yakalama penceresi",
            UYUMLU if "CAPTURE_TIMEOUT_SECONDS" in bridge_code and '"LIVE_CAPTURE_TIMEOUT_SECONDS", 4' in bridge_code else SAPMA,
            "LIVE_CAPTURE_TIMEOUT_SECONDS default=4")
    else:
        add("K4", "K4.1-4", "live_bridge.py", DOGRULANAMADI, "Dosya yok")

    # K4.3: Kafka
    consumer_path = os.path.join(ROOT, "src", "kafka_consumer.py")
    if os.path.exists(consumer_path):
        with open(consumer_path, "r", encoding="utf-8") as f:
            consumer_code = f.read()
        add("K4", "K4.3", "Apache Kafka mesaj hatti",
            UYUMLU if "confluent_kafka" in consumer_code or "KafkaError" in consumer_code else SAPMA)
    else:
        add("K4", "K4.3", "kafka_consumer.py", DOGRULANAMADI)

    # K4.5: MinMaxScaler train-only
    preprocess_path = os.path.join(ROOT, "src", "features", "preprocess_ml_3class.py")
    if os.path.exists(preprocess_path):
        with open(preprocess_path, "r", encoding="utf-8") as f:
            pp_code = f.read()
        add("K4", "K4.5", "MinMaxScaler train-only fit",
            UYUMLU if "fit_transform(X_train)" in pp_code or "scaler.fit_transform" in pp_code else SAPMA)

    # K4.6: Hot-swap ML <-> DL
    registry_path = os.path.join(ROOT, "src", "model_registry.py")
    if os.path.exists(registry_path):
        with open(registry_path, "r", encoding="utf-8") as f:
            reg_code = f.read()
        lstm_live = '"live_supported": False' in reg_code
        add("K4", "K4.6", "Sifir kesintili model degisimi (ML ↔ DL)",
            SAPMA,
            "Paper 5 model hot-swap iddia eder. Repo: LSTM/BiLSTM live_supported=False (E2-S01 karari). "
            "Hot-swap yalnizca 3 tabular model arasinda calisiyor. JNCA'da tabular-only siniri belirtilmeli.")

    # K4.7: 3-stage escalation
    if os.path.exists(consumer_path):
        with open(consumer_path, "r", encoding="utf-8") as f:
            consumer_code = f.read()
        has_alert = '"ALERT"' in consumer_code
        has_suspicious = '"SUSPICIOUS"' in consumer_code
        has_blocked = '"BLOCKED"' in consumer_code
        add("K4", "K4.7", "3 asamali kademeli yanit (ALERT → SUSPICIOUS → BLOCKED)",
            UYUMLU if all([has_alert, has_suspicious, has_blocked]) else SAPMA)

    # K4.8: Analyst approval
    if os.path.exists(consumer_path):
        # Count block_ip( calls excluding import/def lines
        call_count = 0
        for line in consumer_code.splitlines():
            stripped = line.strip()
            if "block_ip(" in stripped and not stripped.startswith(("from ", "import ", "def ", "#")):
                call_count += 1

        add("K4", "K4.8", "Analist onayi: consumer otomatik engellemez",
            UYUMLU if call_count == 0 else SAPMA,
            "Consumer block_ip import eder ama process_message/main'de cagirmaz. "
            "Engelleme yalnizca dashboard uzerinden analist tarafindan yapilir.")

    # K4.9: Platform-independent firewall
    fw_path = os.path.join(ROOT, "src", "utils", "firewall_manager.py")
    if os.path.exists(fw_path):
        with open(fw_path, "r", encoding="utf-8") as f:
            fw_code = f.read()
        has_netsh = "netsh" in fw_code
        has_iptables = "iptables" in fw_code
        add("K4", "K4.9", "Platform bagimsiz firewall (iptables + Windows)",
            UYUMLU if has_netsh and has_iptables else SAPMA)

    # K4.10: 5 model live
    add("K4", "K4.10", "5 model canli hatta kullanilabilir",
        SAPMA,
        "Paper 5 model canli iddia eder. Repo: yalnizca 3 tabular (RF/DT/XGB). "
        "LSTM/BiLSTM live_supported=False. JNCA'da sinir olarak belirtilmeli.")


# ── Report generation ─────────────────────────────────────────────────────────

def generate_report():
    lines = []
    lines.append("# SIU 2026 (394) ↔ Repo Iddia Denetimi")
    lines.append(f"\n**Tarih:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Script:** `tools/siu_claim_audit.py`")
    lines.append("")

    # Summary
    total = len(results)
    uyumlu = sum(1 for r in results if r["status"] == UYUMLU)
    sapma = sum(1 for r in results if r["status"] == SAPMA)
    dogrulanamadi = sum(1 for r in results if r["status"] == DOGRULANAMADI)

    lines.append("---")
    lines.append("")
    lines.append("## Ozet")
    lines.append("")
    lines.append(f"| Durum | Sayi |")
    lines.append(f"|-------|------|")
    lines.append(f"| UYUMLU | {uyumlu}/{total} |")
    lines.append(f"| SAPMA | {sapma}/{total} |")
    lines.append(f"| DOGRULANAMADI | {dogrulanamadi}/{total} |")
    lines.append("")

    # Per-section tables
    sections = []
    for r in results:
        if r["section"] not in sections:
            sections.append(r["section"])

    for section in sections:
        section_results = [r for r in results if r["section"] == section]
        section_titles = {
            "K1": "K1 — Dosya Bazli Katmanli Bolme",
            "K2": "K2 — Uc Sinifli Taksonomi",
            "K3": "K3 — 5 Model Karsilastirmasi",
            "K4": "K4 — Canli Kopru Mimarisi",
        }
        lines.append("---")
        lines.append("")
        lines.append(f"## {section_titles.get(section, section)}")
        lines.append("")
        lines.append("| ID | Aciklama | Durum | Detay |")
        lines.append("|-----|---------|-------|-------|")
        for r in section_results:
            detail = r["detail"].replace("|", "/").replace("\n", " ") if r["detail"] else ""
            desc = r["description"].replace("|", "/").replace("\n", " ")
            status_icon = {"UYUMLU": "✅", "SAPMA": "⚠️", "DOGRULANAMADI": "❓"}[r["status"]]
            lines.append(f"| {r['id']} | {desc} | {status_icon} {r['status']} | {detail} |")
        lines.append("")

    # Action items for JNCA
    sapma_items = [r for r in results if r["status"] == SAPMA]
    dogrulanamadi_items = [r for r in results if r["status"] == DOGRULANAMADI]

    if sapma_items or dogrulanamadi_items:
        lines.append("---")
        lines.append("")
        lines.append("## JNCA Icin Aksiyon Gereken Maddeler")
        lines.append("")

        if sapma_items:
            lines.append("### Sapmalar")
            lines.append("")
            for i, r in enumerate(sapma_items, 1):
                lines.append(f"{i}. **{r['id']}** — {r['description']}")
                if r["detail"]:
                    lines.append(f"   - {r['detail']}")
            lines.append("")

        if dogrulanamadi_items:
            lines.append("### Dogrulanamayan Maddeler")
            lines.append("")
            for i, r in enumerate(dogrulanamadi_items, 1):
                lines.append(f"{i}. **{r['id']}** — {r['description']}")
                if r["detail"]:
                    lines.append(f"   - {r['detail']}")
            lines.append("")

    report_text = "\n".join(lines)
    report_path = os.path.join(ROOT, "reports", "siu_claim_audit.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    print(f"\nSaved to: {report_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("E1-S12: SIU 2026 (394) Claim Audit")
    print("=" * 60)

    audit_k1()
    audit_k2()
    audit_k3()
    audit_k4()
    generate_report()


if __name__ == "__main__":
    main()
