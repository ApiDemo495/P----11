"""Multi-asset testing suite for the prediction app.

Mode A: automated historical backtest of the app's own logic
(engine direction + moves_risk levels) under a strict anti-look-ahead
DataSlice. Mode B: blind bar-by-bar replay for discretionary
filtering, logged to SQLite. Analytics compare raw vs filtered.
"""
