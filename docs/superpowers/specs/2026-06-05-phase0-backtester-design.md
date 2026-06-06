# Phase-0 Backtester Design — Delta-Neutral Funding Capture

**Datum:** 2026-06-05  
**Phase:** 0 — Research & Backtest (kein Live-Handel, kein Paper-Trading)  
**Kapital:** ~€2.000 (~€500–1.000 pro Leg)  
**Exchanges:** Binance, OKX  
**Strategie:** Delta-neutral Funding-Rate Capture auf volatile Altcoins  

---

## 1. Scope & Abgrenzung

Phase 0 liefert ein **Research-Tool**: Coin-Screener + Backtester + Report.  
Phase 1 (Paper-Trading) wird erst nach positivem Phase-0-Ergebnis gebaut.  
Kein Live-Handel ohne explizite menschliche Freigabe (Charter §5).

**Stack:** Python 3.12+, ccxt, pandas, numpy, pyarrow, jinja2, python-dotenv  
**Kein Rust** — Strategie ist nicht latenz-sensitiv; Bottleneck ist Netzwerk-Roundtrip zum Exchange, nicht Code-Laufzeit.  
**Kein ML in Phase 0** — erst regelbasierte Baseline, ML-Layer in Phase 1 optional evaluieren.

---

## 2. Architektur & Verzeichnisstruktur

```
trade4/
├── src/trade4/
│   ├── data/
│   │   ├── binance.py      # Binance-Fetcher (ccxt + REST)
│   │   ├── okx.py          # OKX-Fetcher
│   │   └── store.py        # Parquet-Cache-Management
│   ├── screener/
│   │   └── screener.py     # Funding-Ranking + Liquiditäts-Filter
│   ├── backtest/
│   │   ├── cost_model.py   # Gebühren, Slippage, FDUSD-Flag, Basis-Drift
│   │   └── engine.py       # Vektorisierter Backtest-Engine (pandas)
│   └── report/
│       └── report.py       # HTML-Report + CSV-Export
├── notebooks/
│   └── phase0_research.ipynb   # Interaktive Analyse (ruft src-Module auf)
├── data/                       # Parquet-Cache (gitignored)
├── docs/
├── pyproject.toml
└── .env                        # API-Keys (gitignored)
```

**Kernprinzip:** `src/trade4/` enthält ausschließlich testbare Business-Logik. Das Notebook ist nur die interaktive Schicht — keine Logik drin.

---

## 3. Daten-Fetching & Screener

### Datenquellen

| Datenpunkt | Binance-Endpoint | OKX-Endpoint | Granularität |
|---|---|---|---|
| Funding-Rate-Historie | `/fapi/v1/fundingRate` | `/api/v5/public/funding-rate-history` | 8h, bis 1 Jahr |
| OHLCV | `/fapi/v1/klines` | `/api/v5/market/candles` | 1d |
| Mark-Price | `/fapi/v1/markPriceKlines` | `/api/v5/market/mark-price-candles` | 1h |
| Orderbook-Snapshot | `/fapi/v1/depth` | `/api/v5/market/books` | Top-20-Levels |

API-Keys werden für höhere Rate-Limits genutzt, sind aber für historische Daten nicht zwingend.  
Keys: read-only, **keine Withdraw-Berechtigung** (Charter §10).

### Caching

Erster Lauf: vollständiger Download → Parquet (`data/{exchange}/{type}/{SYMBOL}.parquet`).  
Folgeläufe: Delta-Fetch ab letztem gecachten Timestamp. Verhindert unnötige API-Calls.

### Screener-Logik (zweistufig)

**Stufe 1 — Funding-Filter:**
- `avg_funding_30d` und `avg_funding_90d` pro Coin
- `pct_positive_intervals`: Anteil der Intervalle mit positivem Funding
- Nur Coins mit `avg_funding_30d ≥ entry_threshold` (Default: 0.005%/8h) kommen weiter

**Stufe 2 — Liquiditäts-Filter:**
- Simuliere Order in Höhe der Positionsgröße (~€500) gegen gecachten Orderbook-Levels
- Estimated Slippage ≤ 0.05% pro Leg → bestanden
- Coins wo die Order > 0.5% des 24h-Volumens ausmacht → ausgeschlossen (Charter §2)

**FDUSD-Zero-Fee-Flag:**  
Folgende Coins auf Binance erhalten `spot_fee_bps = 0` (statt 10 bps):  
BTC, ETH, SOL, DOGE, LINK, BNB, XRP (gegen FDUSD, Stand: laufende Promotion "until further notice").  
FDUSD/USDT-Depeg-Risiko wird als separater Kostenpuffer modelliert (0.5 bps pauschal).  
Flag wird im Report als eigene Spalte ausgewiesen.

---

## 4. Backtest Engine & Cost Model

### Cost Model (Charter §6 — hart verdrahtet)

```
round_trip_cost_bps =
    spot_entry_fee  + perp_entry_fee
  + spot_exit_fee   + perp_exit_fee
  + slippage_entry  + slippage_exit
  + basis_drift_expected
  + fdusd_depeg_buffer (0.5 bps wenn FDUSD genutzt)

Net_Edge_bps = expected_funding_cumulative - round_trip_cost_bps

Gate (beide Bedingungen müssen erfüllt sein):
  Net_Edge_bps ≥ 15 bps
  expected_funding_cumulative ≥ 2 × round_trip_cost_bps
```

Default-Fees (Base-Tier, Charter §11):

| Leg | Taker | Maker |
|---|---|---|
| Binance Spot (Standard) | 10 bps | 9 bps |
| Binance Spot (FDUSD) | 0 bps | 0 bps |
| Binance Perp | 5 bps | 2 bps |
| OKX Spot | 10 bps | 8 bps |
| OKX Perp | 5 bps | 2 bps |

### Backtest-Engine (vektorisiert)

**Entry-Signal:**  
`funding_rate[t] ≥ entry_threshold` (Default: 0.005%/8h)  
UND `rolling_avg(funding, last_5_intervals) ≥ 0.003%/8h` (Persistenz-Filter gegen kurzlebige Spikes)

**Exit-Signal:**  
`funding_rate[t] < exit_threshold` (Default: 0.0%/8h) ODER `max_holding_days` (Default: 30) erreicht

**P&L je Zyklus:**  
`Σ(funding_received_per_interval) − round_trip_cost_bps` × Notional

**Delta-Drift-Simulation:**  
Mark-Price-Divergenz (Perp − Spot) wird pro Intervall berechnet.  
Überschreitet Drift ±2%, werden Rebalancing-Kosten (halber Round-Trip) addiert.

### Realismus-Anforderungen (maximale Marktnähe)

1. **Bid/Ask statt Mid-Price:** Käufe auf `ask`, Verkäufe auf `bid`. Spread aus Orderbook — fallback auf `(High−Low)/2` der Candle.
2. **Funding-Intervall-Timing:** Funding wird nur zu exakten Zeitstempeln gutgeschrieben (00:00, 08:00, 16:00 UTC). Backtest rechnet exakt, wie viele Intervalle zwischen Entry und Exit liegen.
3. **Maker-Fill-Wahrscheinlichkeit:** Maker-Fill nur wenn der Markt tatsächlich durch das Limit-Level gehandelt hat. Unfilled → Fallback auf Taker + höhere Gebühr.
4. **Orderbook-basierte Slippage:** Positionsgröße wird Schicht für Schicht durch Orderbook-Levels "gefressen" — kein Pauschalprozentsatz.

### Integrity-Regeln (Charter §11)

- **Walk-Forward:** In-Sample 2023–2024, Out-of-Sample 2025
- **Kein Look-Ahead-Bias:** Entry-Entscheidung nutzt ausschließlich Daten, die zum Zeitpunkt t bereits bekannt waren
- **Base-Tier-Fees als Default** — nie optimistische Annahmen
- **Stress-Test Pflicht:** 2× Fees, 2× Slippage, Funding-Flip-Szenario

---

## 5. Report & Output

### Struktur

**1. Screener-Ergebnisse**  
Tabelle: `Coin | avg_funding_30d | avg_funding_90d | pct_positive | slippage_est_bps | fdusd_zero_fee | net_ev_gate_passed`

**2. Per-Coin Backtest** (nur für gate_passed = True)
- Return-Histogramm (Verteilung der Zyklus-Returns)
- Equity-Kurve (kumulativ, netto nach allen Kosten)
- Drawdown-Chart
- Kosten-Breakdown: `fees | slippage | basis_drift | net_edge`

**3. Sensitivity-Tabelle** (Pflicht §11)  
Matrix: `1×/2× Fees` × `1×/2× Slippage` × `normal/flip-Funding` → Net_Edge_bps (rot wenn < 15 bps)

**4. Walk-Forward-Vergleich**  
In-Sample (2023–2024) vs. Out-of-Sample (2025) side-by-side

**5. Failure-Mode-Summary** (Pflicht für Phase-1-Gate §5)  
Konkret je Szenario: wie verliert diese Strategie Geld?

### Ehrlichkeits-Regeln (Charter §12)

- Alle Zahlen netto nach Kosten, Breakdown sichtbar
- Annualisierung nur wenn Sample ≥ 90 Tage, sonst absolute Zahlen
- Paper-Label auf jeder Report-Seite
- Wenn Out-of-Sample < In-Sample → roter Banner oben im Report
- Keine Cherry-Picking-Fenster

---

## 6. Was Phase 0 NICHT liefert

- Kein laufender Prozess, kein Paper-Trader, kein Scheduler
- Kein ML-Layer (kommt optional in Phase 1)
- Keine Live-Order-Platzierung
- Keine Empfehlung zum Live-Handel — das ist eine menschliche Entscheidung (Charter §5)

---

## 7. Exit-Kriterium für Phase 0 (Charter §5)

Phase 0 ist abgeschlossen wenn:
- ≥ 1 Coin den Net-EV-Gate (§6) besteht
- Sensitivity-Analyse zeigt positive Net_Edge auch bei 2× Kosten
- Walk-Forward Out-of-Sample ist nicht signifikant schlechter als In-Sample
- Failure-Mode-Summary ist dokumentiert

Erst dann beginnt Phase-1-Planung — mit separatem Design-Dokument.
