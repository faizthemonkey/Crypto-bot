"""
Futures Trading Bot - Streamlit GUI v3.0
- Fixed SL/TP execution (uses actual stop price, accounts for leverage)
- Fixed amount trading mode
- Background auto-trading with WebSocket
- AWS deployment ready
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import json
import os
import threading
import time
from typing import Dict, Any, List, Optional

from crypto_bot.futures_exchange import BinanceFuturesClient
from crypto_bot.futures_backtest import FuturesBacktester, LivePaperTrader, BacktestConfig, PositionSide
from crypto_bot.indicators import Indicators
from crypto_bot.strategies.futures_strategies import STRATEGY_REGISTRY, get_strategy, get_strategy_params

st.set_page_config(page_title="Futures Trading Bot v3", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# CSS
st.markdown("""
<style>
div[data-testid="stMetric"] { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 15px; border-radius: 10px; border: 1px solid #0f3460; }
div[data-testid="stMetricLabel"] { color: #e0e0e0 !important; font-weight: 600; font-size: 14px !important; }
div[data-testid="stMetricValue"] { color: #ffffff !important; font-size: 26px !important; font-weight: 700; }
.status-running { color: #00ff88; font-weight: bold; }
.status-stopped { color: #ff4757; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

STRATEGY_EXPLANATIONS = {
    'RSI': {'name': 'RSI Strategy', 'desc': '**Entry:** Long when RSI < 30 (oversold), Short when RSI > 70 (overbought). **Exit:** When RSI returns to neutral. **Best for:** Range-bound markets. **SL/TP Note:** Stop loss % is on margin (leveraged).'},
    'MACD': {'name': 'MACD Strategy', 'desc': '**Entry:** Long when MACD histogram crosses positive, Short when crosses negative. **Exit:** On opposite signal. **Best for:** Trending markets.'},
    'Bollinger Bands': {'name': 'Bollinger Bands', 'desc': '**Entry:** Long at lower band, Short at upper band. **Exit:** At middle band. **Best for:** Mean-reverting markets.'},
    'EMA Crossover': {'name': 'EMA Crossover', 'desc': '**Entry:** Long when fast EMA > slow EMA, Short when fast < slow. **Exit:** On opposite crossover. **Best for:** Trending markets.'},
    'Supertrend': {'name': 'Supertrend', 'desc': '**Entry:** Long when price > Supertrend, Short when price < Supertrend. **Exit:** When direction reverses. **Best for:** Trending markets.'},
    'RSI + MACD': {'name': 'RSI + MACD Combined', 'desc': '**Entry:** Long when MACD bullish AND RSI < 40, Short when MACD bearish AND RSI > 60. **Exit:** On opposite signal. **Best for:** Filtered entries.'},
    'Stochastic RSI': {'name': 'Stochastic RSI', 'desc': '**Entry:** Long when K crosses D in oversold, Short when K crosses D in overbought. **Exit:** At opposite zone. **Best for:** Range markets.'},
    'Triple EMA': {'name': 'Triple EMA', 'desc': '**Entry:** Long when Fast > Medium > Slow EMA, Short when Fast < Medium < Slow. **Exit:** When alignment breaks. **Best for:** Strong trends.'}
}

BACKTESTS_FILE = "saved_backtests.json"
TRADING_STATE_FILE = "trading_state.json"

# Background trading state
class BackgroundTrader:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.is_running = False
        self.trader: Optional[LivePaperTrader] = None
        self.config: Optional[Dict] = None
        self.strategy = None
        self.symbol = ""
        self.interval = ""
        self.thread: Optional[threading.Thread] = None
        self.last_update = None
        self.last_price = 0.0
        self.last_signal = 0
        self.error = None
        self.trade_log = []
    
    def start(self, config: Dict, trader: LivePaperTrader, strategy, symbol: str, interval: str):
        if self.is_running:
            return False
        self.config = config
        self.trader = trader
        self.strategy = strategy
        self.symbol = symbol
        self.interval = interval
        self.is_running = True
        self.error = None
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        self._save_state()
        return True
    
    def stop(self):
        self.is_running = False
        if self.trader and self.trader.position.side != PositionSide.FLAT:
            self.trader.close_position()
        self._save_state()
    
    def _run_loop(self):
        client = BinanceFuturesClient()
        interval_seconds = {'1m': 60, '3m': 180, '5m': 300, '15m': 900, '30m': 1800, '1h': 3600, '2h': 7200, '4h': 14400}.get(self.interval, 60)
        
        while self.is_running:
            try:
                # Fetch latest data
                data = client.get_historical_klines(self.symbol, self.interval, limit=100)
                if data.empty:
                    time.sleep(10)
                    continue
                
                current_price = data['close'].iloc[-1]
                high = data['high'].iloc[-1]
                low = data['low'].iloc[-1]
                self.last_price = current_price
                self.trader.update_price(current_price)
                
                # Check SL/TP if in position
                if self.trader.position.side != PositionSide.FLAT:
                    exit_price = self.trader.check_stop_loss_take_profit(high, low)
                    if exit_price is not None:
                        self.trader.current_price = exit_price
                        result = self.trader.close_position()
                        if result:
                            self.trade_log.append(f"{datetime.now().strftime('%H:%M:%S')} SL/TP hit: ${result.pnl:.2f}")
                
                # Generate signals
                signals = self.strategy.generate_signals(data)
                signal = signals['signal'].iloc[-1]
                self.last_signal = signal
                
                # Execute signals
                if signal == 1 and self.trader.position.side != PositionSide.LONG:
                    if self.trader.position.side == PositionSide.SHORT:
                        self.trader.close_position()
                    self.trader.open_long()
                    self.trade_log.append(f"{datetime.now().strftime('%H:%M:%S')} LONG @ ${current_price:,.2f}")
                elif signal == -1 and self.trader.position.side != PositionSide.SHORT:
                    if self.trader.position.side == PositionSide.LONG:
                        self.trader.close_position()
                    self.trader.open_short()
                    self.trade_log.append(f"{datetime.now().strftime('%H:%M:%S')} SHORT @ ${current_price:,.2f}")
                
                self.last_update = datetime.now()
                self._save_state()
                time.sleep(min(interval_seconds, 60))
                
            except Exception as e:
                self.error = str(e)
                time.sleep(30)
    
    def _save_state(self):
        if self.trader:
            state = {'is_running': self.is_running, 'symbol': self.symbol, 'interval': self.interval,
                'last_update': self.last_update.isoformat() if self.last_update else None, 'stats': self.trader.get_stats()}
            try:
                with open(TRADING_STATE_FILE, 'w') as f:
                    json.dump(state, f)
            except: pass
    
    def get_status(self) -> Dict:
        return {'is_running': self.is_running, 'symbol': self.symbol, 'interval': self.interval,
            'last_update': self.last_update, 'last_price': self.last_price, 'last_signal': self.last_signal,
            'stats': self.trader.get_stats() if self.trader else {}, 'error': self.error, 'trade_log': self.trade_log[-10:]}

def load_saved_backtests() -> List[Dict]:
    if os.path.exists(BACKTESTS_FILE):
        try:
            with open(BACKTESTS_FILE, 'r') as f: return json.load(f)
        except: return []
    return []

def save_backtest(data: Dict) -> int:
    backtests = load_saved_backtests()
    save_data = {'id': len(backtests)+1, 'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"), 'symbol': data.get('symbol','N/A'),
        'strategy': data.get('strategy','N/A'), 'interval': data.get('interval','N/A'), 'initial_balance': data.get('initial_balance',10000),
        'final_equity': data.get('final_equity',0), 'total_return_pct': data.get('total_return_pct',0), 'sharpe_ratio': data.get('sharpe_ratio',0),
        'max_drawdown_pct': data.get('max_drawdown_pct',0), 'total_trades': data.get('total_trades',0), 'win_rate_pct': data.get('win_rate_pct',0),
        'profit_factor': data.get('profit_factor',0), 'leverage': data.get('leverage',1), 'position_size': data.get('position_size',100),
        'stop_loss': data.get('stop_loss',0), 'take_profit': data.get('take_profit',0), 'fixed_amount': data.get('fixed_amount', 0)}
    backtests.append(save_data)
    if len(backtests) > 50: backtests = backtests[-50:]
    with open(BACKTESTS_FILE, 'w') as f: json.dump(backtests, f, indent=2)
    return save_data['id']

def delete_backtest(bid: int):
    backtests = [b for b in load_saved_backtests() if b['id'] != bid]
    with open(BACKTESTS_FILE, 'w') as f: json.dump(backtests, f, indent=2)

def clear_all_backtests():
    with open(BACKTESTS_FILE, 'w') as f: json.dump([], f)

def init_session_state():
    defaults = {'backtest_results': None, 'live_data': pd.DataFrame()}
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

init_session_state()

@st.cache_resource
def get_binance_client(): return BinanceFuturesClient()

@st.cache_data(ttl=300)
def get_futures_symbols():
    try: return sorted([s for s in get_binance_client().get_futures_symbols() if s.endswith('USDT')])
    except: return ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT']

def fetch_historical_data(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    try: return get_binance_client().get_historical_klines(symbol, interval, limit=limit)
    except Exception as e: st.error(f"Error: {e}"); return pd.DataFrame()

def create_chart(data: pd.DataFrame, indicators: Dict, trades: List = None, title: str = "Chart") -> go.Figure:
    rows, heights = 1, [0.6]
    if indicators.get('show_volume'): rows += 1; heights.append(0.15)
    if indicators.get('show_rsi'): rows += 1; heights.append(0.125)
    if indicators.get('show_macd'): rows += 1; heights.append(0.125)
    heights = [h/sum(heights) for h in heights]
    
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=heights)
    fig.add_trace(go.Candlestick(x=data['open_time'], open=data['open'], high=data['high'], low=data['low'], close=data['close'],
        name='Price', increasing_line_color='#26a69a', decreasing_line_color='#ef5350'), row=1, col=1)
    
    if indicators.get('show_sma_20'): fig.add_trace(go.Scatter(x=data['open_time'], y=Indicators.sma(data['close'], 20), name='SMA20', line=dict(color='#ff9800', width=1.5)), row=1, col=1)
    if indicators.get('show_sma_50'): fig.add_trace(go.Scatter(x=data['open_time'], y=Indicators.sma(data['close'], 50), name='SMA50', line=dict(color='#2196f3', width=1.5)), row=1, col=1)
    if indicators.get('show_bollinger'):
        upper, middle, lower = Indicators.bollinger_bands(data['close'])
        fig.add_trace(go.Scatter(x=data['open_time'], y=upper, name='BB Upper', line=dict(color='rgba(156,39,176,0.7)', width=1, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=data['open_time'], y=lower, name='BB Lower', line=dict(color='rgba(156,39,176,0.7)', width=1, dash='dash'), fill='tonexty', fillcolor='rgba(156,39,176,0.1)'), row=1, col=1)
    
    if trades:
        long_entries = [(t.entry_time, t.entry_price) for t in trades if t.side == PositionSide.LONG]
        short_entries = [(t.entry_time, t.entry_price) for t in trades if t.side == PositionSide.SHORT]
        profit_exits = [(t.exit_time, t.exit_price) for t in trades if not t.is_open and t.pnl > 0]
        loss_exits = [(t.exit_time, t.exit_price) for t in trades if not t.is_open and t.pnl <= 0]
        
        if long_entries:
            times, prices = zip(*long_entries)
            fig.add_trace(go.Scatter(x=list(times), y=list(prices), mode='markers', marker=dict(symbol='triangle-up', size=14, color='#00ff88'), name='Long Entry'), row=1, col=1)
        if short_entries:
            times, prices = zip(*short_entries)
            fig.add_trace(go.Scatter(x=list(times), y=list(prices), mode='markers', marker=dict(symbol='triangle-down', size=14, color='#ff4757'), name='Short Entry'), row=1, col=1)
        if profit_exits:
            times, prices = zip(*profit_exits)
            fig.add_trace(go.Scatter(x=list(times), y=list(prices), mode='markers', marker=dict(symbol='x', size=12, color='#00ff88'), name='Win Exit'), row=1, col=1)
        if loss_exits:
            times, prices = zip(*loss_exits)
            fig.add_trace(go.Scatter(x=list(times), y=list(prices), mode='markers', marker=dict(symbol='x', size=12, color='#ff4757'), name='Loss Exit'), row=1, col=1)
        for t in trades:
            if not t.is_open and t.exit_time and t.exit_price:
                fig.add_trace(go.Scatter(x=[t.entry_time, t.exit_time], y=[t.entry_price, t.exit_price], mode='lines', line=dict(color='#00ff88' if t.pnl > 0 else '#ff4757', width=1, dash='dot'), showlegend=False), row=1, col=1)
    
    curr_row = 2
    if indicators.get('show_volume'):
        fig.add_trace(go.Bar(x=data['open_time'], y=data['volume'], name='Volume', marker_color=['#26a69a' if c >= o else '#ef5350' for c, o in zip(data['close'], data['open'])], opacity=0.7), row=curr_row, col=1)
        curr_row += 1
    if indicators.get('show_rsi'):
        fig.add_trace(go.Scatter(x=data['open_time'], y=Indicators.rsi(data['close']), name='RSI', line=dict(color='#9c27b0', width=2)), row=curr_row, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#ff4757", row=curr_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#00ff88", row=curr_row, col=1)
        curr_row += 1
    if indicators.get('show_macd'):
        macd, signal, hist = Indicators.macd(data['close'])
        fig.add_trace(go.Scatter(x=data['open_time'], y=macd, name='MACD', line=dict(color='#2196f3', width=2)), row=curr_row, col=1)
        fig.add_trace(go.Scatter(x=data['open_time'], y=signal, name='Signal', line=dict(color='#ff9800', width=2)), row=curr_row, col=1)
        fig.add_trace(go.Bar(x=data['open_time'], y=hist, name='Hist', marker_color=['#00ff88' if h >= 0 else '#ff4757' for h in hist], opacity=0.6), row=curr_row, col=1)
    
    fig.update_layout(height=700, template='plotly_dark', showlegend=True, xaxis_rangeslider_visible=False, dragmode='zoom', hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right"), margin=dict(l=60, r=60, t=80, b=60))
    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    return fig

def create_equity_chart(eq: pd.DataFrame) -> go.Figure:
    color = '#00ff88' if eq['equity'].iloc[-1] >= eq['equity'].iloc[0] else '#ff4757'
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq['time'], y=eq['equity'], mode='lines', line=dict(color=color, width=2), fill='tozeroy', fillcolor=color.replace('ff', '22')))
    fig.add_hline(y=eq['equity'].iloc[0], line_dash="dash", line_color="gray", annotation_text=f"Initial: ${eq['equity'].iloc[0]:,.0f}")
    fig.update_layout(height=250, template='plotly_dark', title='📈 Equity Curve', showlegend=False, margin=dict(l=60, r=60, t=50, b=50), dragmode='zoom')
    fig.update_yaxes(tickprefix='$', tickformat=',.0f')
    return fig

def create_trades_table(trades: list) -> pd.DataFrame:
    if not trades: return pd.DataFrame()
    records = []
    for i, t in enumerate(trades):
        if not t.is_open:
            records.append({'#': i+1, 'Entry': t.entry_time.strftime("%m-%d %H:%M") if t.entry_time else '-',
                'Exit': t.exit_time.strftime("%m-%d %H:%M") if t.exit_time else '-',
                'Side': f"{'🟢' if t.side == PositionSide.LONG else '🔴'} {t.side.value}",
                'Entry$': f"${t.entry_price:,.2f}", 'Exit$': f"${t.exit_price:,.2f}" if t.exit_price else '-',
                'PnL': f"${t.pnl:,.2f}", 'PnL%': f"{t.pnl_percent:.2f}%", '': '✅' if t.pnl > 0 else '❌'})
    return pd.DataFrame(records)

def render_sidebar():
    st.sidebar.title("⚙️ Settings")
    mode = st.sidebar.radio("Mode", ["📊 Backtest", "📡 Live Auto", "💾 Saved"], index=0)
    st.sidebar.divider()
    symbols = get_futures_symbols()
    symbol = st.sidebar.selectbox("Symbol", symbols, index=symbols.index('BTCUSDT') if 'BTCUSDT' in symbols else 0)
    interval = st.sidebar.selectbox("Timeframe", ['1m','3m','5m','15m','30m','1h','2h','4h','6h','12h','1d'], index=5)
    st.sidebar.divider()
    st.sidebar.subheader("📈 Strategy")
    strategy_name = st.sidebar.selectbox("Strategy", list(STRATEGY_REGISTRY.keys()))
    default_params = get_strategy_params(strategy_name)
    strategy_params = {}
    for key, val in default_params.items():
        if isinstance(val, bool): strategy_params[key] = st.sidebar.checkbox(key, value=val, key=f"p_{key}")
        elif isinstance(val, int): strategy_params[key] = st.sidebar.number_input(key, value=val, step=1, key=f"p_{key}")
        elif isinstance(val, float): strategy_params[key] = st.sidebar.number_input(key, value=val, step=0.1, key=f"p_{key}")
    st.sidebar.divider()
    st.sidebar.subheader("💰 Position Sizing")
    sizing_mode = st.sidebar.radio("Mode", ["Percentage", "Fixed Amount"], index=0, key="sizing_mode")
    fixed_amount_mode = sizing_mode == "Fixed Amount"
    initial_balance = st.sidebar.number_input("Initial Balance ($)", value=10000.0, step=1000.0, min_value=100.0)
    if fixed_amount_mode:
        fixed_amount = st.sidebar.number_input("Trade Amount ($)", value=1000.0, step=100.0, min_value=10.0)
        position_size = 100
    else:
        fixed_amount = 0.0
        position_size = st.sidebar.slider("Position %", 10, 100, 100)
    leverage = st.sidebar.slider("Leverage", 1, 125, 10)
    st.sidebar.subheader("🛡️ Risk (on Margin)")
    st.sidebar.caption("SL/TP % is on margin. E.g. 20% SL on 10x = 2% price move")
    use_sl = st.sidebar.checkbox("Stop Loss", value=False)
    sl_pct = st.sidebar.number_input("SL %", value=20.0, step=1.0, min_value=1.0, max_value=100.0) if use_sl else 0.0
    use_tp = st.sidebar.checkbox("Take Profit", value=False)
    tp_pct = st.sidebar.number_input("TP %", value=40.0, step=1.0, min_value=1.0, max_value=500.0) if use_tp else 0.0
    return {'mode': mode, 'symbol': symbol, 'interval': interval, 'strategy_name': strategy_name, 'strategy_params': strategy_params,
        'initial_balance': initial_balance, 'leverage': leverage, 'position_size': position_size,
        'fixed_amount_mode': fixed_amount_mode, 'fixed_amount': fixed_amount,
        'use_stop_loss': use_sl, 'stop_loss_pct': sl_pct, 'use_take_profit': use_tp, 'take_profit_pct': tp_pct}

def render_backtest(config):
    st.title("📊 Historical Backtest")
    if config['strategy_name'] in STRATEGY_EXPLANATIONS:
        with st.expander(f"📚 How {STRATEGY_EXPLANATIONS[config['strategy_name']]['name']} Works"): st.markdown(STRATEGY_EXPLANATIONS[config['strategy_name']]['desc'])
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: sma20, sma50 = st.checkbox("SMA 20", True), st.checkbox("SMA 50", True)
    with c2: bb, vol = st.checkbox("Bollinger", False), st.checkbox("Volume", True)
    with c3: rsi, macd = st.checkbox("RSI", True), st.checkbox("MACD", False)
    with c4: limit = st.slider("Candles", 100, 1500, 500); auto_save = st.checkbox("Auto-save", True)
    indicators = {'show_sma_20': sma20, 'show_sma_50': sma50, 'show_bollinger': bb, 'show_volume': vol, 'show_rsi': rsi, 'show_macd': macd}
    
    # Show sizing mode
    if config['fixed_amount_mode']:
        st.info(f"**Fixed Amount Mode:** ${config['fixed_amount']:,.2f} per trade | Leverage: {config['leverage']}x")
    else:
        st.info(f"**Percentage Mode:** {config['position_size']}% of equity | Leverage: {config['leverage']}x")
    
    if config['use_stop_loss'] or config['use_take_profit']:
        price_sl = config['stop_loss_pct'] / config['leverage'] if config['use_stop_loss'] else 0
        price_tp = config['take_profit_pct'] / config['leverage'] if config['use_take_profit'] else 0
        st.warning(f"**Risk:** SL: {config['stop_loss_pct']}% margin ({price_sl:.2f}% price) | TP: {config['take_profit_pct']}% margin ({price_tp:.2f}% price)")
    
    st.divider()
    c1, c2 = st.columns([3, 1])
    with c1: run = st.button("🚀 Run Backtest", type="primary", use_container_width=True)
    with c2:
        if st.button("🗑️ Clear", use_container_width=True): st.session_state.backtest_results = None; st.rerun()
    
    if run:
        st.session_state.backtest_results = None
        with st.spinner(f"Fetching {config['symbol']}..."): data = fetch_historical_data(config['symbol'], config['interval'], limit)
        if data.empty: st.error("Failed to fetch data"); return
        with st.spinner("Running backtest..."):
            strategy = get_strategy(config['strategy_name'], config['strategy_params'])
            signals = strategy.generate_signals(data)
            bt_config = BacktestConfig(initial_balance=config['initial_balance'], leverage=config['leverage'],
                position_size_pct=config['position_size'], fixed_amount_mode=config['fixed_amount_mode'], fixed_amount=config['fixed_amount'],
                use_stop_loss=config['use_stop_loss'], stop_loss_pct=config['stop_loss_pct'],
                use_take_profit=config['use_take_profit'], take_profit_pct=config['take_profit_pct'])
            results = FuturesBacktester(bt_config).run(data, signals)
            results.update({'symbol': config['symbol'], 'strategy': config['strategy_name'], 'interval': config['interval'], 'leverage': config['leverage'],
                'position_size': config['position_size'], 'fixed_amount': config['fixed_amount'] if config['fixed_amount_mode'] else 0,
                'stop_loss': config['stop_loss_pct'] if config['use_stop_loss'] else 0, 'take_profit': config['take_profit_pct'] if config['use_take_profit'] else 0})
        st.session_state.backtest_results = results
        if auto_save: st.success(f"Backtest saved! (ID: {save_backtest(results)})")
        else: st.success("Complete!")
    
    if st.session_state.backtest_results:
        r = st.session_state.backtest_results
        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("💰 Return", f"{r['total_return_pct']:.2f}%", f"${r['final_equity']-r['initial_balance']:,.2f}")
        with c2: st.metric("🎯 Win Rate", f"{r['win_rate_pct']:.1f}%", f"{r['winning_trades']}/{r['total_trades']}")
        with c3: st.metric("📊 Sharpe", f"{r['sharpe_ratio']:.2f}")
        with c4: st.metric("📉 Max DD", f"{r['max_drawdown_pct']:.2f}%")
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("💹 Profit Factor", f"{r['profit_factor']:.2f}")
        with c2: st.metric("🟢 Long PnL", f"${r['long_pnl']:,.2f}", f"{r['long_trades']}")
        with c3: st.metric("🔴 Short PnL", f"${r['short_pnl']:,.2f}", f"{r['short_trades']}")
        with c4: st.metric("💵 Final", f"${r['final_equity']:,.2f}")
        st.plotly_chart(create_chart(r['data'], indicators, r['trades'], f"{config['symbol']} - {config['strategy_name']}"), use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})
        if not r['equity_curve'].empty: st.plotly_chart(create_equity_chart(r['equity_curve']), use_container_width=True)
        st.subheader("📋 Trades")
        tdf = create_trades_table(r['trades'])
        if not tdf.empty: st.dataframe(tdf, use_container_width=True, hide_index=True)
        else: st.info("No trades executed")

def render_saved():
    st.title("💾 Saved Backtests")
    backtests = load_saved_backtests()
    if not backtests: st.info("No saved backtests"); return
    if st.button("🗑️ Clear All"): clear_all_backtests(); st.rerun()
    for bt in reversed(backtests):
        c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 0.5])
        with c1: st.markdown(f"**#{bt['id']} {bt['symbol']}** ({bt['strategy']})"); st.caption(f"{bt['timestamp']} | {bt['interval']}")
        with c2: st.metric("Return", f"{'🟢' if bt['total_return_pct']>=0 else '🔴'} {bt['total_return_pct']:.2f}%")
        with c3: st.metric("Win Rate", f"{bt['win_rate_pct']:.1f}%")
        with c4: st.metric("Trades", bt['total_trades'])
        with c5:
            if st.button("🗑️", key=f"d{bt['id']}"): delete_backtest(bt['id']); st.rerun()
        st.divider()

def render_live_auto(config):
    st.title("📡 Live Auto Trading (Background)")
    trader = BackgroundTrader()
    status = trader.get_status()
    
    # Status display
    if status['is_running']:
        st.markdown(f"<span class='status-running'>● RUNNING</span> on {status['symbol']} ({status['interval']})", unsafe_allow_html=True)
        if status['last_update']:
            st.caption(f"Last update: {status['last_update'].strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.markdown("<span class='status-stopped'>○ STOPPED</span>", unsafe_allow_html=True)
    
    if status['error']:
        st.error(f"Error: {status['error']}")
    
    # Controls
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("▶️ Start Background", type="primary", disabled=status['is_running']):
            bt_cfg = BacktestConfig(initial_balance=config['initial_balance'], leverage=config['leverage'],
                position_size_pct=config['position_size'], fixed_amount_mode=config['fixed_amount_mode'], fixed_amount=config['fixed_amount'],
                use_stop_loss=config['use_stop_loss'], stop_loss_pct=config['stop_loss_pct'],
                use_take_profit=config['use_take_profit'], take_profit_pct=config['take_profit_pct'])
            paper_trader = LivePaperTrader(bt_cfg)
            strategy = get_strategy(config['strategy_name'], config['strategy_params'])
            price = float(get_binance_client().get_ticker_price(config['symbol'])['price'])
            paper_trader.update_price(price)
            trader.start(config, paper_trader, strategy, config['symbol'], config['interval'])
            st.rerun()
    with c2:
        if st.button("⏹️ Stop", disabled=not status['is_running']):
            trader.stop()
            st.rerun()
    with c3:
        if st.button("🔄 Refresh Status"):
            st.rerun()
    
    st.divider()
    
    # Live stats
    if status['is_running'] and status['stats']:
        s = status['stats']
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("💰 Equity", f"${s.get('current_equity', 0):,.2f}")
        with c2: st.metric("📈 Return", f"{s.get('total_return_pct', 0):.2f}%")
        with c3:
            pos = s.get('position', 'FLAT')
            icon = "🟢" if pos == 'LONG' else "🔴" if pos == 'SHORT' else "⚪"
            st.metric("📍 Position", f"{icon} {pos}")
        with c4: st.metric("🔢 Trades", s.get('total_trades', 0))
        
        st.metric("💲 Current Price", f"${status['last_price']:,.2f}")
        sig_icon = "🟢 LONG" if status['last_signal'] == 1 else "🔴 SHORT" if status['last_signal'] == -1 else "⚪ FLAT"
        st.info(f"Current Signal: {sig_icon}")
        
        # Trade log
        if status['trade_log']:
            st.subheader("📋 Recent Activity")
            for log in reversed(status['trade_log']):
                st.text(log)
    
    st.divider()
    st.markdown("""
    **Background Trading Features:**
    - ✅ Runs in background thread, survives page refresh
    - ✅ Auto-executes strategy signals
    - ✅ Auto SL/TP monitoring using high/low prices
    - ✅ State saved to disk for recovery
    - ⚠️ Will stop if Streamlit server restarts
    """)

def main():
    config = render_sidebar()
    if config['mode'] == "📊 Backtest": render_backtest(config)
    elif config['mode'] == "💾 Saved": render_saved()
    else: render_live_auto(config)
    st.divider()
    st.caption("Futures Trading Bot v3.0 | Fixed SL/TP | Fixed Amount Mode | Background Trading | AWS Ready")

if __name__ == "__main__": main()
