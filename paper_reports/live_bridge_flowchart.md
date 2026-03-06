# Live Bridge — Real-Time Inference Pipeline

```mermaid
flowchart TD
    A["Network Interface\n(Wi-Fi / Ethernet)"]
    B["Packet Sniffer\nScapy · 4-second windows"]
    C["Flow Feature Extractor\nCICFlowMeter CLI\n78 → 20 selected features"]
    D["Feature Normaliser\nMinMaxScaler\ntrain-fitted · leakage-free"]
    E["Kafka Producer\nlive_bridge.py\nTopic: nids_flows"]
    F["Kafka Broker\nApache Kafka 3.4\nDocker · port 9092"]
    G["AI Inference Consumer\nkafka_consumer.py\nHot-swappable model"]
    H{"Predicted\nClass"}
    I["Class 0 — Benign\nLog · Allow"]
    J["Class 1 — Volumetric\nDDoS / DoS / Botnet\nAlert · Block IP"]
    K["Class 2 — Semantic\nPortScan / WebAttack / BruteForce\nAlert · Block IP"]
    L["SHAP Explainer\nTop-3 feature attribution\nper alert"]
    M["Dashboard & Alert Store\nStreamlit · SQLite\nlocalhost:8501"]

    A --> B --> C --> D --> E --> F --> G --> H
    H -- Benign --> I --> M
    H -- Volumetric --> J --> L --> M
    H -- Semantic  --> K --> L --> M

    style A fill:#1d3557,stroke:#457b9d,color:#f1faee
    style B fill:#1d3557,stroke:#457b9d,color:#f1faee
    style C fill:#1b4332,stroke:#40916c,color:#d8f3dc
    style D fill:#1b4332,stroke:#40916c,color:#d8f3dc
    style E fill:#3d105a,stroke:#9d4edd,color:#f0d9ff
    style F fill:#3d105a,stroke:#9d4edd,color:#f0d9ff
    style G fill:#7f3000,stroke:#e85d04,color:#fff3e0
    style H fill:#212529,stroke:#adb5bd,color:#f8f9fa
    style I fill:#1b4332,stroke:#40916c,color:#d8f3dc
    style J fill:#5c0000,stroke:#c1121f,color:#ffe0e0
    style K fill:#5c0000,stroke:#c1121f,color:#ffe0e0
    style L fill:#0d1b2a,stroke:#778da9,color:#e0e1dd
    style M fill:#0d1b2a,stroke:#778da9,color:#e0e1dd
```

**Figure X.** Live Bridge real-time inference pipeline. Network packets are captured in 4-second windows, converted to 20 bidirectional flow features by CICFlowMeter, normalised by a train-fitted MinMaxScaler, and published to a Kafka topic. The AI Consumer classifies each flow into one of three classes; detected attacks trigger SHAP-attributed alerts and automated IP blocking.
