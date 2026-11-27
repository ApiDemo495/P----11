# The app was publishing orders the exchange rejects

**Segment 15 · 2026-11-27**

## The standing rule this breaks

> "the app must publish an order actually executable at the user's broker."

It did not. `app/trading.py` built every order like this:

```python
qty = round(notional / entry, 6)
"stopPrice": round(float(_sl_price), 2)
```

Binance USDⓈ-M futures enforces three per-symbol filters:

| filter | rule |
|---|---|
| `LOT_SIZE` | `quantity % stepSize == 0`, `quantity >= minQty` |
| `PRICE_FILTER` | `price % tickSize == 0` |
| `MIN_NOTIONAL` | `price × quantity >= minNotional` |

Measured filters (2026-11-27):

| symbol | tickSize | stepSize | minQty | minNotional |
|---|---|---|---|---|
| BTCUSDT | 0.1 | 0.001 | 0.001 | 100.0 |
| ETHUSDT | 0.01 | 0.001 | 0.001 | 20.0 |
| BNBUSDT | 0.01 | 0.010 | 0.010 | 20.0 |
| PAXGUSDT | 0.01 | 0.001 | 0.001 | 20.0 |

At the app's then-default `max_order_usdt = 25.0`, with BTC at 68273.5:

```
qty      round(25 / 68273.5, 6) = 0.000366   not a multiple of 0.001  REJECT
stop     round(68273.5×0.99, 2) = 67590.76   not a multiple of 0.1    REJECT
notional 25 USDT                             below the 100 floor      REJECT
```

**Three of three filters failed.** The entry is rejected, so no trade
happens; the protective stop is rejected too, so a position that did open
would sit unbracketed. ETH/BNB/PAXG cleared their 20 USDT floor but still
failed `LOT_SIZE`, because `round(25/price, 6)` is not a multiple of the
0.001 step either. The bug was never one filter — neither quantity nor
price was ever made filter-aware.

## The fix

New `app/exchange_filters.py`:

- `Filters` — the three constraints, with `floor_step` / `floor_tick` that
  **round down** (documented best practice: an order may only shrink below
  the risk already approved, never grow past it).
- `refresh()` — pulls live `/fapi/v1/exchangeInfo` with a 1-hour cache and
  falls back to the measured `BUNDLED` table offline, so a cold start
  still produces a legal order instead of a crash.
- `legal_order()` — rounds quantity down to the step and prices down to
  the tick, correcting stop and target **away from the entry** (a long's
  stop rounds down, its target up), so rounding never tightens risk after
  sizing.
- `min_order_usdt()` — the *effective* floor, which is higher than the raw
  filter because quantity must also be a step multiple:

  ```
  ceil_pad( minNotional / price / stepSize ) × stepSize × price
  ```

  BTC is the clean case: minNotional 100 USDT with stepSize 0.001 at
  ~68000 a coin needs **0.002 BTC = 136 USDT**. Quoting "the minimum is
  100" is true of the filter and false of the order.

`Trader.execute()` now:

1. sizes the notional as before (still never levered up),
2. runs it through `legal_order()`,
3. **refuses with a numbered message** if no legal order exists, instead
   of sending a guaranteed rejection:

```
order would be rejected by the exchange: quantity rounds to 0 at
stepSize 0.001 (notional too small for BTCUSDT) (max_order_usdt must be
at least 136 for BTCUSDT)
```

`DEFAULTS["max_order_usdt"]` moves **25 → 150**, which clears BTC's 136
floor with room for price drift — verified against the live table by a
test, so the default cannot silently fall below the floor again.

## Verification

Orders actually sent, end to end through `execute()`:

| symbol @ 150 USDT | qty | notional | SL | TP |
|---|---|---|---|---|
| BTC | 0.002 | 136.55 | 67863.8 | 69092.8 |
| ETH | 0.061 | 149.45 | 2435.3 | 2479.4 |
| BNB | 0.34 | 149.41 | 436.81 | 444.73 |
| PAXG | 0.04 | 149.40 | 3712.59 | 3779.82 |

Every quantity is a step multiple, every price a tick multiple, every
notional over the floor. Below the floor the order is refused, not sent.

`GET /api/trading/status` now carries the `exchange` block: per-symbol
filters, the effective floor in USDT, and which symbols the current budget
covers. The position-size control shows that verdict live instead of
offering a number the broker will bounce.

Tests: `tests/test_exchange_filters.py` (18). Full suite **1273 passed,
3 skipped**.

## Note on the one changed fixture

`tests/test_trading_profiles.py` used `entry=100.0` with a 25 USDT budget
— a placeholder-sized order that the new layer correctly refuses. Its
intent ("no path bypasses the gates; three brackets are sent") is
unchanged; the fixture is now a real BTC order at 68273.5.
