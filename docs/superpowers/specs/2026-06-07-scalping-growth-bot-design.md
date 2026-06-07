# Small Account Growth Bot — Design Spec
**Date:** 2026-06-07  
**Status:** Approved by user  
**Goal:** Exponentielles Kapitalwachstum aus 2.000 € Startkapital via algorithmischem Scalping auf Binance Futures Perpetuals.

---

## 1. Ziel & Rahmenbedingungen

| Parameter | Wert |
|---|---|
| Startkapital | 2.000 € |
| Exchange | Binance Futures (USDT-M Perpetuals) |
| Max. Risiko pro Trade | 2% des aktuellen Kontokapitals |
| Max. Tagesverlust (Circuit Breaker) | −3% des Tagesstartsaldo |
| Max. offene Positionen | 2 (gemeinsamer Pool beider Strategien) |
| Deployment | VPS (Linux, Python 3.11+) |
| Phasen | Phase 1: Backtest/Validierung — Phase 2: Live-Execution |

---

## 2. System-Architektur

```
┌─────────────────────────────────────────────────────┐
│                   DATA LAYER                        │
│  Binance Futures REST (Klines) + WebSocket (Live)   │
│  Reuse: trade4/data/binance.py                      │
└────────────────────┬────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
┌────────▼────────┐   ┌──────────▼──────────┐
│  MODUL A        │   │  MODUL B             │
│  EMA-Cross      │   │  Pump/Spike Scanner  │
│  Momentum       │   │                      │
└────────┬────────┘   └──────────┬───────────┘
         └───────────┬───────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│                 RISK MANAGER                        │
│  • Positionsgröße (2% Risiko)                       │
│  • Liquidation Buffer Check                         │
│  • Daily Loss Limit (−3%)                           │
│  • Max. 2 offene Positionen                         │
│  • Korrelations-Guard (kein BTC+ETH Long = 2x same) │
│  • Funding Rate Guard (trade4 reuse)                │
│  • Session Filter (EU + US Window)                  │
└────────────────────┬────────────────────────────────┘
                     │
          ┌──────────┴──────────┐
          │                     │
┌─────────▼──────────┐  ┌──────▼──────────────────────┐
│   PHASE 1           │  │   PHASE 2                   │
│   BACKTEST ENGINE   │  │   EXECUTION ENGINE          │
│   trade4 reuse      │  │   Binance Futures API       │
└─────────────────────┘  └─────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│           STATE / PERSISTENCE (SQLite)              │
│  Open Positions, Daily PnL, Trade Log               │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│              MONITORING (Telegram + Log)            │
│  Trade Open/Close, Daily Summary, Circuit Breaker   │
│  Reuse: trade4/report/ (Equity-Kurve, HTML-Report)  │
└─────────────────────────────────────────────────────┘
```

---

## 3. Asset-Selektion (Dynamischer Screener)

Kein festes Asset-Set. Täglicher Screener (trade4-Screener-Logik adaptiert) filtert Binance Futures Symbole nach:

- **Mindestliquidität:** 24h-Volumen > 200 Mio. USDT
- **Volatilitäts-Gate:** ATR(14) auf 1h-Chart > Mindest-Schwellwert (dynamisch kalibriert)
- **Blacklist:** Symbole ohne echten Spot-Markt (z.B. BTCDOMUSDT — bereits in trade4 bekannt)
- **Top-N Auswahl:** Top 20 Symbole nach ATR-Score

Typische Kandidaten: BTC/USDT, ETH/USDT, SOL/USDT, DOGE/USDT, WIF/USDT, PEPE/USDT, BNB/USDT, XRP/USDT

---

## 4. Signal-Logik

### 4.1 Modul A — EMA-Cross Momentum

**Timeframes:** 1m (Signal), 15m (Trend-Filter)

**Entry Long:**
1. EMA9 kreuzt EMA21 von unten auf 1m-Chart (Golden Cross)
2. RSI(14) > 50
3. Preis > EMA200 auf 15m-Chart (Bullish Bias)
4. ATR(14) auf 1m > Mindestschwelle (kein flacher Markt; Schwellwert wird in Backtest-Phase kalibriert, Startpunkt: 0,05% des Asset-Preises)
5. Session-Filter aktiv
6. Funding Rate Guard: kein extrem positives Funding (Longs bereits überfüllt)

**Entry Short:** Spiegelbildlich.

**Exit — Stop-Loss:** `Entry ± 1,5 × ATR(14)` auf 1m, dynamisch.

**Exit — Take-Profit:** Trailing-Stop: aktiviert sobald Preis 1× ATR im Profit, zieht bei 0,5× ATR nach.

**Max. Haltedauer:** 30 Minuten — danach Market-Close, egal ob im Profit oder nicht.

**Positionsgröße:**
```
Risiko_EUR   = USDT_Gesamtguthaben (Futures Account) × 0,02
SL_Abstand   = 1,5 × ATR(14)   [in USDT/Coin]
Qty          = Risiko_EUR / SL_Abstand
Notional     = Qty × Preis
Hebel        = Notional / Margin_Allocated  → Deckel: 20x
```

---

### 4.2 Modul B — Pump/Spike Scanner

**Timeframe:** 1m (Echtzeit-WebSocket)

**Trigger Long:**
1. Volumen der letzten geschlossenen 1m-Kerze > 3× gleitender 20-Perioden-Durchschnitt des Symbols
2. Preisbewegung (Close − Open) > +1,5% in dieser Kerze
3. Symbol aktuell nicht im offenen Positions-Pool
4. Session-Filter aktiv

**Entry:** Market-Order (Priorität: Schnelligkeit)

**Exit — Stop-Loss:** Fest −0,8% vom Entry-Preis, sofort als Stop-Market-Order platziert.

**Exit — Take-Profit:** +1,2% vom Entry-Preis, sofort als Limit-Order platziert.

**Hebel:** 7x fest (niedrig, da Pumps schnell umkehren können).

**Max. Haltedauer:** 5 Minuten — danach Market-Close.

---

## 5. Risk Manager

### 5.1 Positionsgrößen-Kalkulation
Läuft vor jedem Order-Submit. Inputs: aktuelles Kapital, ATR, Entry-Preis. Output: Qty, Hebel.

### 5.2 Liquidation Buffer Check
Vor jedem Order: berechne Liquidationspreis (Binance-Formel für USDT-M Perpetuals).  
Bedingung: `Stop-Loss-Preis` muss mindestens 20% Abstand zum Liquidationspreis haben.  
Wenn nicht erfüllt → Hebel reduzieren bis Bedingung gilt, oder Trade ablehnen.

### 5.3 Daily Loss Circuit Breaker
- Tagesstart-Kapital wird um 00:00 UTC aus SQLite geladen.
- Wenn `aktueller_PnL < −3% × Tagesstart-Kapital` → alle offenen Positionen schließen, keine neuen Orders bis 00:00 UTC.

### 5.4 Session-Filter
- **Aktive Handelszeiten:** 07:00–11:00 UTC (EU-Open) + 13:00–21:00 UTC (US-Session)
- Außerhalb: keine neuen Entries. Offene Positionen werden weitergemanagt.

### 5.5 Funding Rate Guard
- Reuse: trade4-Funding-Daten (Binance Futures API, 8h-Intervall)
- Schwellwert: Funding Rate > +0,15% → keine neuen Long-Entries auf diesem Symbol
- Schwellwert: Funding Rate < −0,15% → keine neuen Short-Entries auf diesem Symbol

### 5.6 Korrelations-Guard
- Max. 1 Position aus der Gruppe {BTC, ETH} gleichzeitig (Korrelation > 0,8 in der Regel)
- Gilt nicht für uncorrelated Assets (z.B. DOGE, WIF)

---

## 6. State & Persistence (SQLite)

Tabellen:
- `positions`: symbol, side, entry_price, qty, leverage, sl_price, tp_price, opened_at, strategy
- `trades`: alle abgeschlossenen Trades mit PnL, Dauer, Strategie
- `daily_summary`: Datum, Tagesstart-Kapital, realisierter PnL, Anzahl Trades, Max-DD

Bei Bot-Neustart: offene Positionen aus `positions` laden und in Runtime-State hydrieren.

---

## 7. Monitoring (Telegram)

Events die einen Alert auslösen:
- Trade geöffnet (Symbol, Side, Entry, SL, TP, Hebel, Strategie)
- Trade geschlossen (PnL, Dauer, Grund: SL / TP / Timeout / Manuell)
- Circuit Breaker ausgelöst
- Tägliche Zusammenfassung (08:00 UTC): Trades, PnL, aktuelles Kapital, Equity-Kurve PNG

---

## 8. Backtest-Plan (Phase 1)

**Datengrundlage:** Binance Futures Kline-Daten (1m + 15m), mindestens 12 Monate historisch.

**Validierungsstrategie:**
- In-Sample: Monate 1–9 → Parameteroptimierung (EMA-Längen, RSI-Level, ATR-Multiplier, Pump-Threshold)
- Out-of-Sample: Monate 10–12 → Validierung ohne Anpassung (Walk-Forward)

**Mindestanforderungen für Phase-2-Freigabe:**
- Sharpe Ratio ≥ 1,5 (annualisiert)
- Max. Drawdown ≤ 20%
- Win Rate ≥ 45% (realisiertes durchschnittliches RRR ≥ 1:1,5 über alle Trades)
- Profit Factor ≥ 1,4
- Out-of-Sample-Performance nicht mehr als 30% schlechter als In-Sample

**Kostenmodell:** trade4-Kostenmodell (Maker 0,02%, Taker 0,05%, Slippage-Estimate).

---

## 9. Projektstruktur

```
trade4/
├── src/trade4/
│   ├── data/           # Reuse (Binance fetcher)
│   ├── screener/       # Reuse + erweitern (Volatilitäts-Score)
│   ├── backtest/       # Reuse (Engine + Kostenmodell)
│   ├── report/         # Reuse
│   └── scalper/        # NEU
│       ├── signals/
│       │   ├── ema_cross.py      # Modul A
│       │   └── pump_scanner.py   # Modul B
│       ├── risk_manager.py
│       ├── state.py              # SQLite persistence
│       ├── executor.py           # Phase 2: Live Order Submit
│       ├── telegram_bot.py
│       └── main.py               # Entry point
```

---

## 10. Phasenplan

### Phase 1 — Backtest & Validierung
1. Asset-Screener auf Volatilitäts-Score erweitern
2. Modul A Signal-Logik implementieren + backtesten
3. Modul B Signal-Logik implementieren + backtesten
4. Walk-Forward Validation durchführen
5. Report: Equity-Kurven, Sensitivity-Tabelle beider Module

### Phase 2 — Live-Execution (nach bestandener Validierung)
1. Binance Futures API-Keys einrichten (Testnet zuerst)
2. Execution Engine + State Persistence implementieren
3. Telegram Monitoring aktivieren
4. Papiergeld-Test 2 Wochen auf Testnet
5. Live-Start mit echtem Kapital

---

## 11. Kritische Risiken

| Risiko | Mitigation |
|---|---|
| Overfitting im Backtest | Walk-Forward Validation, Out-of-Sample-Gate |
| Liquidations-Gap | Liquidation Buffer Check vor jedem Order |
| Verlust-Serien (10+ in Folge) | Circuit Breaker (−3%/Tag), 2% Risiko-Deckel |
| API-Ausfall während offener Position | State Persistence + Reconnect-Logik + Telegram-Alert |
| Pump-Reversal (Modul B) | Fester 0,8% SL, max. 7x Hebel, 5min Timeout |
| Psychologischer Druck bei Drawdown | Vollautomatisch — kein manuelles Eingreifen im Betrieb |
