flowchart LR

A[Ağ Arayüzü\nCanlı Trafik] --> B[Scapy Dinleyici\nPaket Yakalama]
B --> C[CICFlowMeter\nAkış Özelliği Çıkarma\n78 → 20 özellik]
C --> D[Apache Kafka\nMesaj Kuyruğu]

D --> E[Yapay Zeka Tüketici\nModel Tahmini\nKarar Ağacı · Rastgele Orman · XGBoost · LSTM · BiLSTM]

E --> F{Saldırı\nTespit\nEdildi?}

F -- Evet --> G[Güvenlik Duvarı\nEngeli]
F -- Hayır --> H[Streamlit Paneli\nGerçek Zamanlı\nİzleme]
