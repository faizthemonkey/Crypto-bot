import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from typing import Optional

import matplotlib
matplotlib.use("TkAgg")  # Use TkAgg backend for embedding in Tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import mplfinance as mpf
import pandas as pd

from .config import Config
from .backtest import Backtester
from .strategies import MovingAverageCrossover
from .data_loader import load_ohlcv_csv


class BacktestGUI(tk.Tk):
    """Simple GUI for local backtesting with candlestick visualization."""

    def __init__(self, config: Optional[Config] = None):
        super().__init__()
        self.title("Crypto Bot Backtester")
        self.geometry("1100x700")

        self.config_obj = config or Config()
        self.backtester = Backtester(self.config_obj)

        self.file_path_var = tk.StringVar()
        self.fast_period_var = tk.IntVar(value=10)
        self.slow_period_var = tk.IntVar(value=30)
        self.initial_balance_var = tk.DoubleVar(value=10000.0)

        self._build_layout()

    def _build_layout(self) -> None:
        control_frame = ttk.Frame(self)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        # File selection
        ttk.Label(control_frame, text="Historical CSV:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        entry_file = ttk.Entry(control_frame, textvariable=self.file_path_var, width=70)
        entry_file.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Button(control_frame, text="Browse", command=self._browse_file).grid(row=0, column=2, padx=5, pady=2)

        # Strategy params
        ttk.Label(control_frame, text="Fast MA:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(control_frame, textvariable=self.fast_period_var, width=8).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(control_frame, text="Slow MA:").grid(row=1, column=1, sticky=tk.E, padx=120, pady=2)
        slow_entry = ttk.Entry(control_frame, textvariable=self.slow_period_var, width=8)
        slow_entry.grid(row=1, column=1, sticky=tk.W, padx=180, pady=2)

        ttk.Label(control_frame, text="Initial Balance:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(control_frame, textvariable=self.initial_balance_var, width=10).grid(row=1, column=3, sticky=tk.W, padx=5, pady=2)

        ttk.Button(control_frame, text="Run Backtest", command=self._run_backtest).grid(row=0, column=3, padx=10, pady=2)

        # Metrics frame
        self.metrics_text = tk.Text(self, height=6, width=120)
        self.metrics_text.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        self.metrics_text.config(state=tk.DISABLED)

        # Plot frame
        plot_frame = ttk.Frame(self)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.figure = Figure(figsize=(10, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _browse_file(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select historical OHLCV CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if file_path:
            self.file_path_var.set(file_path)

    def _run_backtest(self) -> None:
        path = self.file_path_var.get().strip()
        if not path:
            messagebox.showwarning("Missing file", "Please select a historical CSV file first.")
            return

        try:
            df = load_ohlcv_csv(path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error loading data", str(exc))
            return

        fast = self.fast_period_var.get()
        slow = self.slow_period_var.get()
        initial_balance = self.initial_balance_var.get()

        if fast <= 0 or slow <= 0 or fast >= slow:
            messagebox.showwarning("Invalid parameters", "Fast MA must be > 0, Slow MA > 0, and Fast < Slow.")
            return

        strategy = MovingAverageCrossover(fast_period=fast, slow_period=slow)

        try:
            results = self.backtester.run_backtest_on_data(
                strategy=strategy,
                data=df,
                initial_balance=initial_balance,
            )
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Backtest error", str(exc))
            return

        self._show_metrics(results)
        self._plot_results(results["data"])

    def _show_metrics(self, results: dict) -> None:
        summary = {
            "Total Return (%)": results.get("total_return", 0) * 100,
            "Annualized Return (%)": results.get("annualized_return", 0) * 100,
            "Annualized Volatility": results.get("annualized_volatility", 0),
            "Sharpe Ratio": results.get("sharpe_ratio", 0),
            "Max Drawdown (%)": results.get("max_drawdown", 0) * 100,
            "Number of Trades": results.get("num_trades", 0),
            "Win Rate (%)": results.get("win_rate", 0) * 100,
        }

        lines = ["Backtest Summary:"]
        for k, v in summary.items():
            if isinstance(v, float):
                lines.append(f"- {k}: {v:.2f}")
            else:
                lines.append(f"- {k}: {v}")

        self.metrics_text.config(state=tk.NORMAL)
        self.metrics_text.delete("1.0", tk.END)
        self.metrics_text.insert(tk.END, "\n".join(lines))
        self.metrics_text.config(state=tk.DISABLED)

    def _plot_results(self, data: pd.DataFrame) -> None:
        if "open_time" not in data.columns:
            messagebox.showwarning("Plot warning", "Data is missing 'open_time' column; cannot plot candlesticks.")
            return

        df = data.copy()
        df.set_index("open_time", inplace=True)

        # Prepare OHLC data for mplfinance
        ohlc = df[["open", "high", "low", "close"]].rename(
            columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"}
        )
        if "volume" in df.columns:
            ohlc["Volume"] = df["volume"]

        # Markers for buy/sell signals
        buys = df[df.get("positions", 0) > 0]
        sells = df[df.get("positions", 0) < 0]

        self.figure.clf()
        ax_price = self.figure.add_subplot(2, 1, 1)
        ax_equity = self.figure.add_subplot(2, 1, 2, sharex=ax_price)

        apds = []
        if not buys.empty:
            apds.append(
                mpf.make_addplot(
                    buys["close"],
                    type="scatter",
                    markersize=50,
                    marker="^",
                    color="g",
                    ax=ax_price,
                )
            )
        if not sells.empty:
            apds.append(
                mpf.make_addplot(
                    sells["close"],
                    type="scatter",
                    markersize=50,
                    marker="v",
                    color="r",
                    ax=ax_price,
                )
            )

        mpf.plot(
            ohlc,
            type="candle",
            style="yahoo",
            ax=ax_price,
            addplot=apds if apds else None,
            volume=False,
            show_nontrading=False,
        )

        if "equity_curve" in df.columns:
            ax_equity.plot(df.index, df["equity_curve"], color="blue")
            ax_equity.set_ylabel("Equity")

        ax_price.set_ylabel("Price")

        self.figure.tight_layout()
        self.canvas.draw()


def main() -> None:
    gui = BacktestGUI()
    gui.mainloop()


if __name__ == "__main__":
    main()
