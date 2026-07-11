"""Monitoring: Telegram notifications, equity tracking and the weekly report.

These components give the otherwise headless executor visibility:
  - notifier.py  — Telegram sender + a logging handler that pushes ERROR logs as alerts.
  - equity.py    — periodic portfolio valuation appended to a time-series for the dashboard.
  - reporter.py  — weekly performance report (positions + P&L) pushed to Telegram.
"""
