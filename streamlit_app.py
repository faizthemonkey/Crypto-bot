"""
Futures Trading Bot - Streamlit GUI v2.0
Enhanced with strategy explanations, trade markers, saved backtests, and improved UI
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import json
import os
from typing import Dict, Any, List

from crypto_bot.futures_exchange import BinanceFuturesClient
from crypto_bot.futures_backtest import FuturesBacktester, LivePaperTrader, BacktestConfig, PositionSide
from crypto_bot.indicators import Indicators
from crypto_bot.strategies.futures_strategies import STRATEGY_REGISTRY, get_strategy, get_strategy_params

st.set_page_config(page_title="Futures Trading Bot", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# Enhanced CSS for better contrast
st.markdown("""
<style>
div[data-testid="stMetric"] { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 15px; border-radius: 10px; border: 1px solid #0f3460; }
div[data-testid="stMetricLabel"] { color: #e0e0e0 !important; font-weight: 600; font-size: 14px !important; }
div[data-testid="stMetricValue"] { color: #ffffff !important; font-size: 26px !important; font-weight: 700; }
.strategy-box { background: linear-gradient(135deg, #1e3a5f 0%, #0d2137 100%); padding: 20px; border-radius: 10px; border-left: 4px solid #00d4ff; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# Strategy explanations
STRATEGY_EXPLANATIONS = {
    'RSI': {'name': 'RSI Strategy', 'desc': '**Entry:** Long when RSI < 30 (oversold), Short when RSI > 70 (overbought). **Exit:** When RSI returns to neutral. **Best for:** Range-bound markets.'},
    'MACD': {'name': 'MACD Strategy', 'desc': '**Entry:** Long when MACD histogram crosses positive, Short when crosses negative. **Exit:** On opposite signal. **Best for:** Trending markets.'},
    'Bollinger Bands': {'name': 'Bollinger Bands', 'desc': '**Entry:** Long at lower band, Short at upper band. **Exit:** At middle band. **Best for:** Mean-reverting markets.'},
    'EMA Crossover': {'name': 'EMA Crossover', 'desc': '**Entry:** Long when fast EMA > slow EMA, Short when fast < slow. **Exit:** On opposite crossover. **Best for:** Trending markets.'},
    'Supertrend': {'name': 'Supertrend', 'desc': '**Entry:** Long when price > Supertrend, Short when price < Supertrend. **Exit:** When direction reverses. **Best for:** Trending markets.'},
    'RSI + MACD': {'name': 'RSI + MACD Combined', 'desc': '**Entry:** Long when MACD bullish AND RSI < 40, Short when MACD bearish AND RSI > 60. **Exit:** On opposite signal. **Best for:** Filtered entries.'},
    'Stochastic RSI': {'name': 'Stochastic RSI', 'desc': '**Entry:** Long when K crosses D in oversold, Short when K crosses D in overbought. **Exit:** At opposite zone. **Best for:** Range markets.'},
    'Triple EMA': {'name': 'Triple EMA', 'desc': '**Entry:** Long when Fast > Medium > Slow EMA, Short when Fast < Medium < Slow. **Exit:** When alignment breaks. **Best for:** Strong trends.'}
}

BACKTESTS_FILE = "saved_backtests.json"

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
        'stop_loss': data.get('stop_loss',0), 'take_profit': data.get('take_profit',0)}
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
    for key, val in {'backtest_results': None, 'paper_trader': None, 'is_live_trading': False, 'live_data': pd.DataFrame()}.items():
        if key not in st.session_state: st.session_state[key] = val

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
    if indicators.get('show_ema_12'): fig.add_trace(go.Scatter(x=data['open_time'], y=Indicators.ema(data['close'], 12), name='EMA12', line=dict(color='#ffeb3b', width=1.5)), row=1, col=1)
    if indicators.get('show_ema_26'): fig.add_trace(go.Scatter(x=data['open_time'], y=Indicators.ema(data['close'], 26), name='EMA26', line=dict(color='#9c27b0', width=1.5)), row=1, col=1)
    
    if indicators.get('show_bollinger'):
        upper, middle, lower = Indicators.bollinger_bands(data['close'])
        fig.add_trace(go.Scatter(x=data['open_time'], y=upper, name='BB Upper', line=dict(color='rgba(156,39,176,0.7)', width=1, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=data['open_time'], y=lower, name='BB Lower', line=dict(color='rgba(156,39,176,0.7)', width=1, dash='dash'), fill='tonexty', fillcolor='rgba(156,39,176,0.1)'), row=1, col=1)
    
    # Trade markers - entries and exits
    if trades:
        long_entries = [(t.entry_time, t.entry_price) for t in trades if t.side == PositionSide.LONG]
        short_entries = [(t.entry_time, t.entry_price) for t in trades if t.side == PositionSide.SHORT]
        profit_exits = [(t.exit_time, t.exit_price) for t in trades if not t.is_open and t.pnl > 0]
        loss_exits = [(t.exit_time, t.exit_price) for t in trades if not t.is_open and t.pnl <= 0]
        
        if long_entries:
            times, prices = zip(*long_entries)
            fig.add_trace(go.Scatter(x=list(times), y=list(prices), mode='markers', marker=dict(symbol='triangle-up', size=14, color='#00ff88', line=dict(color='white', width=1)), name='Long Entry', hovertemplate='LONG ENTRY<br>$%{y:,.2f}<extra></extra>'), row=1, col=1)
        if short_entries:
            times, prices = zip(*short_entries)
            fig.add_trace(go.Scatter(x=list(times), y=list(prices), mode='markers', marker=dict(symbol='triangle-down', size=14, color='#ff4757', line=dict(color='white', width=1)), name='Short Entry', hovertemplate='SHORT ENTRY<br>$%{y:,.2f}<extra></extra>'), row=1, col=1)
        if profit_exits:
            times, prices = zip(*profit_exits)
            fig.add_trace(go.Scatter(x=list(times), y=list(prices), mode='markers', marker=dict(symbol='x', size=12, color='#00ff88', line=dict(width=3)), name='Win Exit', hovertemplate='WIN EXIT<br>$%{y:,.2f}<extra></extra>'), row=1, col=1)
        if loss_exits:
            times, prices = zip(*loss_exits)
            fig.add_trace(go.Scatter(x=list(times), y=list(prices), mode='markers', marker=dict(symbol='x', size=12, color='#ff4757', line=dict(width=3)), name='Loss Exit', hovertemplate='LOSS EXIT<br>$%{y:,.2f}<extra></extra>'), row=1, col=1)
        
        # Connect entry to exit with lines
        for t in trades:
            if not t.is_open and t.exit_time and t.exit_price:
                color = '#00ff88' if t.pnl > 0 else '#ff4757'
                fig.add_trace(go.Scatter(x=[t.entry_time, t.exit_time], y=[t.entry_price, t.exit_price], mode='lines', line=dict(color=color, width=1, dash='dot'), showlegend=False, hoverinfo='skip'), row=1, col=1)
    
    curr_row = 2
    if indicators.get('show_volume'):
        colors = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(data['close'], data['open'])]
        fig.add_trace(go.Bar(x=data['open_time'], y=data['volume'], name='Volume', marker_color=colors, opacity=0.7), row=curr_row, col=1)
        curr_row += 1
    
    if indicators.get('show_rsi'):
        rsi = Indicators.rsi(data['close'])
        fig.add_trace(go.Scatter(x=data['open_time'], y=rsi, name='RSI', line=dict(color='#9c27b0', width=2)), row=curr_row, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(255,71,87,0.1)", line_width=0, row=curr_row, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor="rgba(0,255,136,0.1)", line_width=0, row=curr_row, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#ff4757", row=curr_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#00ff88", row=curr_row, col=1)
        curr_row += 1
    
    if indicators.get('show_macd'):
        macd, signal, hist = Indicators.macd(data['close'])
        fig.add_trace(go.Scatter(x=data['open_time'], y=macd, name='MACD', line=dict(color='#2196f3', width=2)), row=curr_row, col=1)
        fig.add_trace(go.Scatter(x=data['open_time'], y=signal, name='Signal', line=dict(color='#ff9800', width=2)), row=curr_row, col=1)
        fig.add_trace(go.Bar(x=data['open_time'], y=hist, name='Hist', marker_color=['#00ff88' if h >= 0 else '#ff4757' for h in hist], opacity=0.6), row=curr_row, col=1)
    
    fig.update_layout(height=750, template='plotly_dark', showlegend=True, xaxis_rangeslider_visible=False, dragmode='zoom', hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right", bgcolor='rgba(0,0,0,0.5)'),
        margin=dict(l=60, r=60, t=80, b=60), paper_bgcolor='#0e1117', plot_bgcolor='#0e1117')
    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)', showspikes=True, spikecolor="gray", spikethickness=1)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)', showspikes=True, spikecolor="gray", spikethickness=1)
    return fig

def create_equity_chart(eq: pd.DataFrame) -> go.Figure:
    color = '#00ff88' if eq['equity'].iloc[-1] >= eq['equity'].iloc[0] else '#ff4757'
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq['time'], y=eq['equity'], mode='lines', line=dict(color=color, width=2), fill='tozeroy', fillcolor=color.replace('ff', '22')))
    fig.add_hline(y=eq['equity'].iloc[0], line_dash="dash", line_color="gray", annotation_text=f"Initial: ${eq['equity'].iloc[0]:,.0f}")
    fig.update_layout(height=300, template='plotly_dark', title='📈 Equity Curve', showlegend=False, margin=dict(l=60, r=60, t=50, b=50), dragmode='zoom')
    fig.update_yaxes(tickprefix='$', tickformat=',.0f')
    return fig

def create_trades_table(trades: list) -> pd.DataFrame:
    if not trades: return pd.DataFrame()
    records = []
    for i, t in enumerate(trades):
        if not t.is_open:
            records.append({'#': i+1, 'Entry': t.entry_time.strftime("%m-%d %H:%M") if t.entry_time else 'N/A',
                'Exit': t.exit_time.strftime("%m-%d %H:%M") if t.exit_time else 'N/A',
                'Side': f"{'🟢' if t.side == PositionSide.LONG else '🔴'} {t.side.value}",
                'Entry $': f"${t.entry_price:,.2f}", 'Exit $': f"${t.exit_price:,.2f}" if t.exit_price else 'N/A',
                'PnL': f"${t.pnl:,.2f}", 'PnL%': f"{t.pnl_percent:.2f}%", 'Result': '✅' if t.pnl > 0 else '❌'})
    return pd.DataFrame(records)

def render_sidebar():
    st.sidebar.title("⚙️ Settings")
    mode = st.sidebar.radio("Mode", ["📊 Backtest", "📡 Live Paper", "💾 Saved"], index=0)
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
    st.sidebar.subheader("💰 Config")
    initial_balance = st.sidebar.number_input("Balance ($)", value=10000.0, step=1000.0, min_value=100.0)
    leverage = st.sidebar.slider("Leverage", 1, 125, 10)
    position_size = st.sidebar.slider("Position %", 10, 100, 100)
    st.sidebar.subheader("🛡️ Risk")
    use_sl = st.sidebar.checkbox("Stop Loss", value=False)
    sl_pct = st.sidebar.number_input("SL %", value=2.0, step=0.5) if use_sl else 0.0
    use_tp = st.sidebar.checkbox("Take Profit", value=False)
    tp_pct = st.sidebar.number_input("TP %", value=4.0, step=0.5) if use_tp else 0.0
    return {'mode': mode, 'symbol': symbol, 'interval': interval, 'strategy_name': strategy_name, 'strategy_params': strategy_params,
        'initial_balance': initial_balance, 'leverage': leverage, 'position_size': position_size,
        'use_stop_loss': use_sl, 'stop_loss_pct': sl_pct, 'use_take_profit': use_tp, 'take_profit_pct': tp_pct}

def render_backtest(config):
    st.title("📊 Historical Backtest")
    if config['strategy_name'] in STRATEGY_EXPLANATIONS:
        info = STRATEGY_EXPLANATIONS[config['strategy_name']]
        with st.expander(f"📚 How {info['name']} Works", expanded=False): st.markdown(info['desc'])
    
    st.subheader("📈 Chart Indicators")
    c1, c2, c3, c4 = st.columns(4)
    with c1: sma20, sma50 = st.checkbox("SMA 20", True), st.checkbox("SMA 50", True)
    with c2: ema12, ema26 = st.checkbox("EMA 12", False), st.checkbox("EMA 26", False)
    with c3: bb, vol = st.checkbox("Bollinger", False), st.checkbox("Volume", True)
    with c4: rsi, macd = st.checkbox("RSI", True), st.checkbox("MACD", False)
    indicators = {'show_sma_20': sma20, 'show_sma_50': sma50, 'show_ema_12': ema12, 'show_ema_26': ema26, 'show_bollinger': bb, 'show_volume': vol, 'show_rsi': rsi, 'show_macd': macd}
    
    c1, c2 = st.columns([2, 1])
    with c1: limit = st.slider("Candles", 100, 1500, 500)
    with c2: auto_save = st.checkbox("Auto-save", True)
    
    st.markdown("**Markers:** 🔺 Long Entry | 🔻 Short Entry | ✕ Green = Win Exit | ✕ Red = Loss Exit")
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
            bt_config = BacktestConfig(initial_balance=config['initial_balance'], leverage=config['leverage'], position_size_pct=config['position_size'],
                use_stop_loss=config['use_stop_loss'], stop_loss_pct=config['stop_loss_pct'], use_take_profit=config['use_take_profit'], take_profit_pct=config['take_profit_pct'])
            results = FuturesBacktester(bt_config).run(data, signals)
            results.update({'symbol': config['symbol'], 'strategy': config['strategy_name'], 'interval': config['interval'], 'leverage': config['leverage'],
                'position_size': config['position_size'], 'stop_loss': config['stop_loss_pct'] if config['use_stop_loss'] else 0, 'take_profit': config['take_profit_pct'] if config['use_take_profit'] else 0})
        st.session_state.backtest_results = results
        if auto_save: bid = save_backtest(results); st.success(f"Backtest saved! (ID: {bid})")
        else: st.success("Backtest complete!")
    
    if st.session_state.backtest_results:
        r = st.session_state.backtest_results
        st.divider()
        st.markdown(f"**Settings:** {config['symbol']} | {config['interval']} | {config['strategy_name']} | {config['leverage']}x | {config['position_size']}% | SL: {config['stop_loss_pct'] if config['use_stop_loss'] else 'Off'}% | TP: {config['take_profit_pct'] if config['use_take_profit'] else 'Off'}%")
        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("💰 Return", f"{r['total_return_pct']:.2f}%", f"${r['final_equity']-r['initial_balance']:,.2f}")
        with c2: st.metric("🎯 Win Rate", f"{r['win_rate_pct']:.1f}%", f"{r['winning_trades']}/{r['total_trades']}")
        with c3: st.metric("📊 Sharpe", f"{r['sharpe_ratio']:.2f}", "Good" if r['sharpe_ratio'] > 1 else "Low")
        with c4: st.metric("📉 Max DD", f"{r['max_drawdown_pct']:.2f}%")
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("💹 Profit Factor", f"{r['profit_factor']:.2f}")
        with c2: st.metric("🟢 Long PnL", f"${r['long_pnl']:,.2f}", f"{r['long_trades']} trades")
        with c3: st.metric("🔴 Short PnL", f"${r['short_pnl']:,.2f}", f"{r['short_trades']} trades")
        with c4: st.metric("💵 Final", f"${r['final_equity']:,.2f}")
        st.divider()
        st.plotly_chart(create_chart(r['data'], indicators, r['trades'], f"{config['symbol']} - {config['strategy_name']}"), use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True, 'displaylogo': False})
        if not r['equity_curve'].empty: st.plotly_chart(create_equity_chart(r['equity_curve']), use_container_width=True, config={'scrollZoom': True})
        st.subheader("📋 Trades")
        tdf = create_trades_table(r['trades'])
        if not tdf.empty: st.dataframe(tdf, use_container_width=True, hide_index=True)
        else: st.info("No trades")
        with st.expander("📊 Details"):
            c1, c2, c3 = st.columns(3)
            with c1: st.write(f"**Returns**\n- Total: {r['total_return_pct']:.2f}%\n- Annual: {r['annualized_return_pct']:.2f}%\n- Avg Trade: ${r['avg_trade_pnl']:.2f}")
            with c2: st.write(f"**Risk**\n- Sharpe: {r['sharpe_ratio']:.2f}\n- Sortino: {r['sortino_ratio']:.2f}\n- Calmar: {r['calmar_ratio']:.2f}")
            with c3: st.write(f"**Stats**\n- Avg Win: ${r['avg_win']:.2f}\n- Avg Loss: ${r['avg_loss']:.2f}\n- Consec Wins: {r['max_consecutive_wins']}")

def render_saved():
    st.title("💾 Saved Backtests")
    backtests = load_saved_backtests()
    if not backtests: st.info("No saved backtests"); return
    if st.button("🗑️ Clear All"): clear_all_backtests(); st.rerun()
    st.divider()
    for bt in reversed(backtests):
        c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 0.5])
        with c1: st.markdown(f"**#{bt['id']} {bt['symbol']}** ({bt['strategy']})"); st.caption(f"{bt['timestamp']} | {bt['interval']}")
        with c2: st.metric("Return", f"{'🟢' if bt['total_return_pct']>=0 else '🔴'} {bt['total_return_pct']:.2f}%")
        with c3: st.metric("Win Rate", f"{bt['win_rate_pct']:.1f}%")
        with c4: st.metric("Trades", bt['total_trades'])
        with c5:
            if st.button("🗑️", key=f"d{bt['id']}"): delete_backtest(bt['id']); st.rerun()
        with st.expander("Details"):
            st.write(f"Balance: ${bt['initial_balance']:,.2f} → ${bt['final_equity']:,.2f} | Leverage: {bt['leverage']}x | Sharpe: {bt['sharpe_ratio']:.2f} | DD: {bt['max_drawdown_pct']:.2f}%")
        st.divider()

def render_live(config):
    st.title("📡 Live Paper Trading")
    if config['strategy_name'] in STRATEGY_EXPLANATIONS:
        with st.expander(f"📚 Strategy Info"): st.markdown(STRATEGY_EXPLANATIONS[config['strategy_name']]['desc'])
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("▶️ Start", type="primary", disabled=st.session_state.is_live_trading):
            bt_cfg = BacktestConfig(initial_balance=config['initial_balance'], leverage=config['leverage'], position_size_pct=config['position_size'],
                use_stop_loss=config['use_stop_loss'], stop_loss_pct=config['stop_loss_pct'], use_take_profit=config['use_take_profit'], take_profit_pct=config['take_profit_pct'])
            st.session_state.paper_trader = LivePaperTrader(bt_cfg)
            st.session_state.is_live_trading = True
            price = float(get_binance_client().get_ticker_price(config['symbol'])['price'])
            st.session_state.paper_trader.update_price(price)
            st.rerun()
    with c2:
        if st.button("⏹️ Stop", disabled=not st.session_state.is_live_trading):
            st.session_state.is_live_trading = False
            if st.session_state.paper_trader and st.session_state.paper_trader.position.side != PositionSide.FLAT:
                st.session_state.paper_trader.close_position()
            st.rerun()
    with c3:
        if st.button("🔄 Reset", disabled=st.session_state.is_live_trading):
            st.session_state.paper_trader = None; st.session_state.live_data = pd.DataFrame(); st.rerun()
    
    st.markdown(f"**Config:** {config['symbol']} | {config['interval']} | {config['strategy_name']} | {config['leverage']}x")
    st.divider()
    
    if st.session_state.paper_trader:
        s = st.session_state.paper_trader.get_stats()
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("💰 Equity", f"${s['current_equity']:,.2f}"); st.metric("📈 Return", f"{s['total_return_pct']:.2f}%")
        with c2: st.metric("💵 Balance", f"${s['balance']:,.2f}"); st.metric("📊 Unrealized", f"${s['unrealized_pnl']:,.2f}")
        with c3: pos_icon = "🟢" if s['position']=='LONG' else "🔴" if s['position']=='SHORT' else "⚪"; st.metric("📍 Position", f"{pos_icon} {s['position']}")
        with c4: st.metric("🔢 Trades", s['total_trades']); st.metric("🎯 Win Rate", f"{s['win_rate']:.1f}%")
    
    st.divider()
    indicators = {'show_sma_20': True, 'show_sma_50': True, 'show_bollinger': False, 'show_rsi': True, 'show_volume': True, 'show_macd': False, 'show_ema_12': False, 'show_ema_26': False}
    
    if st.button("🔄 Refresh Data", type="primary"):
        st.session_state.live_data = fetch_historical_data(config['symbol'], config['interval'], 200)
    
    if not st.session_state.live_data.empty:
        data = st.session_state.live_data
        if st.session_state.paper_trader: st.session_state.paper_trader.update_price(data['close'].iloc[-1])
        strategy = get_strategy(config['strategy_name'], config['strategy_params'])
        signals = strategy.generate_signals(data)
        trades = st.session_state.paper_trader.trades if st.session_state.paper_trader else []
        st.plotly_chart(create_chart(data, indicators, trades, f"{config['symbol']} Live"), use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})
        sig = signals['signal'].iloc[-1]
        st.info(f"Signal: {'🟢 LONG' if sig==1 else '🔴 SHORT' if sig==-1 else '⚪ FLAT'}")
    
    st.subheader("Manual Trading")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("📈 Long", use_container_width=True) and st.session_state.paper_trader:
            if st.session_state.paper_trader.position.side == PositionSide.SHORT: st.session_state.paper_trader.close_position()
            st.session_state.paper_trader.open_long(); st.success("LONG opened")
    with c2:
        if st.button("📉 Short", use_container_width=True) and st.session_state.paper_trader:
            if st.session_state.paper_trader.position.side == PositionSide.LONG: st.session_state.paper_trader.close_position()
            st.session_state.paper_trader.open_short(); st.success("SHORT opened")
    with c3:
        if st.button("❌ Close", use_container_width=True) and st.session_state.paper_trader:
            r = st.session_state.paper_trader.close_position()
            if r: st.success(f"Closed: ${r.pnl:.2f}")
            else: st.info("No position")
    
    if st.session_state.paper_trader:
        tdf = create_trades_table(st.session_state.paper_trader.trades)
        if not tdf.empty: st.dataframe(tdf, use_container_width=True, hide_index=True)

def main():
    config = render_sidebar()
    if config['mode'] == "📊 Backtest": render_backtest(config)
    elif config['mode'] == "💾 Saved": render_saved()
    else: render_live(config)
    st.divider()
    st.caption("Futures Trading Bot v2.0 | Binance Futures API | Paper Trading Only")

if __name__ == "__main__": main()
