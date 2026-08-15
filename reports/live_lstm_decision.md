# Canli LSTM Kapsam Karari

**Gorev:** E2-S01 | **Tarih:** 2026-08-15 | **Karar:** Tabular-only siniri

---

## Karar

Canli cikarim hatti **yalnizca tabular modelleri** (Random Forest, Decision Tree, XGBoost) destekler. Sequence modeller (LSTM, BiLSTM) offline benchmark'ta basarili sonuclar verir ancak canli hatta kullanilmaz.

## Gerekce

### Teknik bosluklar

LSTM/BiLSTM canli cikarimda kullanilmak icin asagidaki bosluklar giderilmelidir:

1. **Pencere birikimi yok:** Kafka consumer her mesaji bagimsiz isler. LSTM `(10, 20)` girisi bekler — 10 ardisik akis kaydinin biriktirilmesi gerekir. Mevcut consumer'da bu mekanizma yoktur.

2. **Yanlis reshape:** Consumer'daki Keras yolu `(1, 1, 20)` seklinde reshape yapar. LSTM mimarisi `(1, 10, 20)` bekler. Tek adimlik giris bilimsel olarak anlamsiz sonuc uretir.

3. **Consumer guard eksikligi:** `check_and_reload_model()` fonksiyonu `live_supported` flag'ini kontrol etmiyordu. Dashboard'dan LSTM secildiginde consumer modeli yukleyip hatali cikarim yapabiliyordu. **Bu bosluk bu gorevde kapatilmistir.**

### Zaman baskisi

Buffer implementasyonu tahmini L efor (1-2 hafta). JNCA journal extension timeline'i icinde bu ek is riski yukselten bir faktordur. Tabular modeller canli hatta zaten iyi calisir:
- XGBoost macro-F1: 0.9587
- RF macro-F1: 0.9725
- DT macro-F1: 0.9721

### SIU K4 tutarliligi

SIU 2026 bildirisinde K4 iddiasi ("Canli Kopru mimarisi calisiyor") **tabular modellere dayanir**. Sequence model iddiasi yapilmamistir. Tabular-only siniri K4 ile celismez.

## Uygulanan degisiklikler

| Dosya | Degisiklik |
|-------|-----------|
| `src/kafka_consumer.py` | `load_model_and_scaler()`: non-live model icin default'a fallback |
| `src/kafka_consumer.py` | `check_and_reload_model()`: non-live model switch'i reddedilir |
| `src/dashboard/app.py` | Non-live model seciminde `active_model.txt` yazilmaz |
| `src/dashboard/app.py` | Uyari mesaji guclendirildi; badge "Sadece Offline" gosterir |

## Gelecek calisma

Buffer implementasyonu (E2-S02, P1) planlanmistir:
- IP bazli `deque` buffer (son 10 akis kaydi)
- Pencere dolunca LSTM cikarimi
- Buffer dolana kadar tabular fallback (hibrit mod)
- Latency olcumu (buffer bekleme + cikarim)

## Paper sinir cumlesi (JNCA)

> The live inference pipeline currently supports tabular models (RF, DT, XGBoost)
> processing individual flow records via Kafka. Sequence models (LSTM, BiLSTM),
> which achieve the highest offline macro-F1 (BiLSTM: 97.72%), require a sliding
> window buffer of 10 consecutive flows; this is planned as future work. The
> tabular-only constraint does not invalidate the live architecture claims, as
> the Kafka-based bridge, hot-swap mechanism, and escalation logic operate
> identically regardless of model type.
