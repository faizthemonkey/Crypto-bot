from pathlib import Path

from crypto_bot.config import Config
from crypto_bot.backtest import Backtester
from crypto_bot.strategies import MovingAverageCrossover
from crypto_bot.data_loader import load_ohlcv_csv


def main() -> None:
    # Path to your local CSV (edit this)
    csv_path = Path("data/btcusdt_1h.csv")

    config = Config()
    backtester = Backtester(config)
    strategy = MovingAverageCrossover(fast_period=10, slow_period=30)

    df = load_ohlcv_csv(csv_path)
    results = backtester.run_backtest_on_data(
        strategy=strategy,
        data=df,
        initial_balance=10_000.0,
    )

    summary = backtester.get_summary()
    print("Backtest summary:")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"- {k}: {v:.2f}")
        else:
            print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
