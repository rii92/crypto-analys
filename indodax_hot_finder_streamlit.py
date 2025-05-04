import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class IndodaxHotFinder:
    def __init__(self):
        self.base_url = "https://indodax.com"

    def get_klines(self, symbol, interval='60', limit=100):
        """Get OHLC data from Indodax"""
        try:
            end_time = int(time.time())
            if interval == '1D':
                start_time = end_time - (86400 * limit)
            elif interval == '1W':
                start_time = end_time - (604800 * limit)
            else:
                start_time = end_time - (int(interval) * 60 * limit)

            params = {
                'symbol': symbol.upper(),
                'tf': interval,
                'from': start_time,
                'to': end_time
            }

            response = requests.get(f"{self.base_url}/tradingview/history_v2", params=params)
            data = response.json()

            df = pd.DataFrame(data)
            df = df.rename(columns={
                'Time': 'timestamp',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])

            return df

        except Exception as e:
            st.error(f"Error getting OHLC data for {symbol}: {str(e)}")
            return None

    # Other methods remain the same as in the original class
    # ...

    def get_ticker_all(self):
        """Get all tickers from Indodax"""
        try:
            response = requests.get(f"{self.base_url}/api/ticker_all")
            data = response.json()
            return data.get('tickers', {})
        except Exception as e:
            st.error(f"Error getting tickers: {str(e)}")
            return None

    def get_pairs(self):
        """Get all available trading pairs"""
        try:
            response = requests.get(f"{self.base_url}/api/pairs")
            return response.json()
        except Exception as e:
            st.error(f"Error getting pairs: {str(e)}")
            return None

    def calculate_indicators(self, df):
        """Calculate technical indicators and price levels"""
        try:
            # Same implementation as original
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

            df['MA7'] = df['close'].rolling(window=7).mean()
            df['MA25'] = df['close'].rolling(window=25).mean()
            df['MA99'] = df['close'].rolling(window=99).mean()

            df['BB_middle'] = df['close'].rolling(window=20).mean()
            df['BB_std'] = df['close'].rolling(window=20).std()
            df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
            df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']

            df['HL'] = df['high'] - df['low']
            df['HC'] = abs(df['high'] - df['close'].shift(1))
            df['LC'] = abs(df['low'] - df['close'].shift(1))
            df['TR'] = df[['HL', 'HC', 'LC']].max(axis=1)
            df['ATR'] = df['TR'].rolling(window=14).mean()

            df['rolling_min'] = df['low'].rolling(window=20).min()
            df['rolling_max'] = df['high'].rolling(window=20).max()

            typical_price = (df['high'] + df['low'] + df['close']) / 3
            df['pivot'] = typical_price
            df['r1'] = 2 * df['pivot'] - df['low']
            df['r2'] = df['pivot'] + (df['high'] - df['low'])
            df['s1'] = 2 * df['pivot'] - df['high']
            df['s2'] = df['pivot'] - (df['high'] - df['low'])

            df['volatility'] = (df['ATR'] / df['close']) * 100

            return df

        except Exception as e:
            st.error(f"Error calculating indicators: {str(e)}")
            return df

    def get_price_levels(self, df, current_price):
        """Calculate important price levels"""
        # Same implementation as original
        # ...
        try:
            latest = df.iloc[-1]

            entry_levels = {
                'aggressive': current_price,
                'ideal': min(latest['s1'], current_price * 0.98),
                'conservative': min(latest['s2'], current_price * 0.95)
            }

            support_levels = {
                'strong': latest['s2'],
                'moderate': latest['s1'],
                'dynamic': latest['BB_lower']
            }

            resistance_levels = {
                'first': latest['r1'],
                'second': latest['r2'],
                'dynamic': latest['BB_upper']
            }

            return {
                'entry': entry_levels,
                'support': support_levels,
                'resistance': resistance_levels,
                'atr': latest['ATR'],
                'volatility': latest['volatility']
            }

        except Exception as e:
            st.error(f"Error calculating price levels: {str(e)}")
            return None

    def calculate_targets(self, current_price, atr, score, interval):
        """Calculate stop loss and target levels"""
        # Same implementation as original
        # ...
        try:
            risk_multiplier = 1.0
            if score > 0.8:
                risk_multiplier = 1.5
            elif score > 0.6:
                risk_multiplier = 1.2

            if interval in ['1', '15', '30']:
                target_multipliers = [1.5, 2.5, 4.0]
            elif interval in ['60', '240']:
                target_multipliers = [2.0, 3.5, 5.0]
            else:
                target_multipliers = [3.0, 5.0, 8.0]

            targets = [
                current_price + (atr * mult * risk_multiplier)
                for mult in target_multipliers
            ]

            stop_loss = current_price - (2 * atr)

            return {
                'stop_loss': stop_loss,
                'targets': targets,
                'risk_reward': (targets[1] - current_price) / (current_price - stop_loss)
            }

        except Exception as e:
            st.error(f"Error calculating targets: {str(e)}")
            return None

    def get_bitcoin_comparison(self, symbol, interval='60', limit=100):
        """Get Bitcoin price data for correlation analysis"""
        # Same implementation as original
        # ...
        try:
            btc_symbol = 'btc_idr'
            btc_df = self.get_klines(btc_symbol, interval, limit)
            alt_df = self.get_klines(symbol, interval, limit)

            if btc_df is None or alt_df is None:
                return None

            merged_df = pd.merge(
                alt_df[['timestamp', 'close']],
                btc_df[['timestamp', 'close']],
                on='timestamp',
                suffixes=('_alt', '_btc')
            )

            return merged_df

        except Exception as e:
            st.error(f"Error getting Bitcoin comparison data: {str(e)}")
            return None

    def plot_technical_analysis_plotly(self, symbol, df, price_levels, targets):
        """Plot comprehensive technical analysis using Plotly"""
        try:
            # Create subplots
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=(
                    f'{symbol.upper()} Technical Analysis',
                    'RSI',
                    'MACD',
                    'Volume'
                ),
                row_heights=[0.4, 0.2, 0.2, 0.2]
            )

            # Price Chart with Bollinger Bands and MAs
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['close'], name='Price', line=dict(color='blue')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['MA7'], name='MA7', line=dict(color='orange')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['MA25'], name='MA25', line=dict(color='green')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['BB_upper'], name='BB Upper', line=dict(color='gray', width=0.5)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['BB_lower'], name='BB Lower', line=dict(color='gray', width=0.5),
                          fill='tonexty', fillcolor='rgba(128, 128, 128, 0.2)'),
                row=1, col=1
            )
            fig.add_hline(y=price_levels['entry']['aggressive'], line_dash="dash", line_color="green",
                         annotation_text="Current Price", row=1, col=1)

            # RSI
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['RSI'], name='RSI', line=dict(color='purple')),
                row=2, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

            # MACD
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['MACD'], name='MACD', line=dict(color='blue')),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['Signal'], name='Signal', line=dict(color='orange')),
                row=3, col=1
            )
            fig.add_trace(
                go.Bar(x=df['timestamp'], y=df['MACD'] - df['Signal'], name='Histogram', marker=dict(color='gray', opacity=0.3)),
                row=3, col=1
            )

            # Volume
            fig.add_trace(
                go.Bar(x=df['timestamp'], y=df['volume'], name='Volume', marker=dict(color='gray', opacity=0.5)),
                row=4, col=1
            )

            # Update layout
            fig.update_layout(
                height=800,
                showlegend=False,
                title_text=f"{symbol.upper()} Technical Analysis",
                xaxis4_title="Date",
                yaxis_title="Price (IDR)",
                yaxis2_title="RSI",
                yaxis3_title="MACD",
                yaxis4_title="Volume"
            )

            return fig

        except Exception as e:
            st.error(f"Error plotting technical analysis: {str(e)}")
            return None

    def plot_correlation_plotly(self, symbol, interval):
        """Plot price correlation with Bitcoin using Plotly"""
        try:
            comparison_df = self.get_bitcoin_comparison(symbol, interval)
            if comparison_df is None or comparison_df.empty:
                return None

            correlation = comparison_df['close_alt'].corr(comparison_df['close_btc'])

            # Normalize prices for better comparison
            comparison_df['close_alt_norm'] = comparison_df['close_alt'] / comparison_df['close_alt'].iloc[0]
            comparison_df['close_btc_norm'] = comparison_df['close_btc'] / comparison_df['close_btc'].iloc[0]

            # Create figure with two subplots
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=(
                    f'Price Trend Comparison: {symbol.upper()} vs BTC/IDR (Correlation: {correlation:.3f})',
                    'Correlation Scatter Plot'
                ),
                row_heights=[0.7, 0.3]
            )

            # Price comparison (normalized)
            fig.add_trace(
                go.Scatter(x=comparison_df['timestamp'], y=comparison_df['close_alt_norm'],
                          name=symbol.upper(), line=dict(color='blue')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=comparison_df['timestamp'], y=comparison_df['close_btc_norm'],
                          name='BTC/IDR', line=dict(color='orange')),
                row=1, col=1
            )

            # Correlation scatter plot
            fig.add_trace(
                go.Scatter(x=comparison_df['close_btc'], y=comparison_df['close_alt'],
                          mode='markers', name='Correlation',
                          marker=dict(color='purple', size=8, opacity=0.6)),
                row=2, col=1
            )

            # Update layout
            fig.update_layout(
                height=600,
                title_text=f"{symbol.upper()} vs Bitcoin Correlation Analysis",
                xaxis2_title="BTC Price (IDR)",
                yaxis_title="Normalized Price",
                yaxis2_title=f"{symbol.upper()} Price (IDR)"
            )

            return fig

        except Exception as e:
            st.error(f"Error plotting correlation: {str(e)}")
            return None

    def find_hot_coins(self, interval='60', min_volume_idr=1000000000):
        """Find hot coins at specific timeframe"""
        # Same implementation as original with progress bar for Streamlit
        try:
            hot_coins = []
            pairs = self.get_pairs()
            tickers = self.get_ticker_all()

            if not pairs or not tickers:
                return []

            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, pair in enumerate(pairs):
                try:
                    # Update progress
                    progress = i / len(pairs)
                    progress_bar.progress(progress)
                    status_text.text(f"Analyzing {i+1}/{len(pairs)}: {pair.get('symbol', 'unknown')}")
                    
                    pair_id = pair['ticker_id']

                    if pair_id not in tickers:
                        continue

                    ticker_data = tickers[pair_id]

                    required_fields = ['vol_idr', 'last', 'high', 'low']
                    if not all(field in ticker_data for field in required_fields) or \
                       any(not ticker_data[field] for field in required_fields):
                        continue

                    volume_24h = float(ticker_data['vol_idr'])
                    if volume_24h < min_volume_idr:
                        continue

                    df = self.get_klines(pair['symbol'], interval=interval)
                    if df is None or df.empty:
                        continue

                    df = self.calculate_indicators(df)

                    current_price = float(ticker_data['last'])
                    latest = df.iloc[-1]

                    price_change = ((current_price - df['open'].iloc[0]) / df['open'].iloc[0]) * 100
                    recent_volume = df['volume'].iloc[-1]
                    avg_volume = df['volume'].mean()
                    volume_surge = recent_volume / avg_volume if avg_volume > 0 else 1

                    signals = []
                    if latest['MACD'] > latest['Signal'] and latest['MACD'] > 0:
                        signals.append(f'🟢 MACD Bullish ({interval})')
                    elif latest['MACD'] < latest['Signal'] and latest['MACD'] < 0:
                        signals.append(f'🔴 MACD Bearish ({interval})')

                    if volume_surge > 2:
                        signals.append(f'🚀 Massive Volume ({volume_surge:.1f}x)')
                    elif volume_surge > 1.3:
                        signals.append(f'💪 Strong Volume ({volume_surge:.1f}x)')
                    elif volume_surge < 0.7:
                        signals.append(f'📉 Volume Dropping ({volume_surge:.1f}x)')

                    if latest['RSI'] < 30:
                        signals.append('💫 Oversold (RSI)')
                    elif latest['RSI'] > 70:
                        signals.append('⚠️ Overbought (RSI)')
                    elif 45 <= latest['RSI'] <= 55:
                        signals.append('✨ Optimal RSI Zone')

                    if latest['MA7'] > latest['MA25'] and latest['MA25'] > latest['MA99']:
                        signals.append('📈 Strong Uptrend')
                    elif latest['MA7'] > latest['MA25']:
                        signals.append('📈 Short-term Uptrend')
                    elif latest['MA7'] < latest['MA25'] and latest['MA25'] < latest['MA99']:
                        signals.append('📉 Strong Downtrend')
                    elif latest['MA7'] < latest['MA25']:
                        signals.append('📉 Short-term Downtrend')

                    bb_width = (latest['BB_upper'] - latest['BB_lower']) / latest['BB_middle']
                    if bb_width < 0.03:
                        signals.append('🎯 Bollinger Squeeze')

                    score_components = {
                        'trend': 1 if latest['MA7'] > latest['MA25'] else 0,
                        'momentum': 1 if latest['MACD'] > latest['Signal'] else 0,
                        'volume': min(volume_surge / 2, 1),
                        'rsi': 1 - abs(50 - latest['RSI']) / 50
                    }

                    potential_score = (
                        score_components['trend'] * 0.3 +
                        score_components['momentum'] * 0.3 +
                        score_components['volume'] * 0.2 +
                        score_components['rsi'] * 0.2
                    )

                    sell_components = {
                        'trend': 1 if latest['MA7'] < latest['MA25'] else 0,
                        'momentum': 1 if latest['MACD'] < latest['Signal'] else 0,
                        'volume': min(1 / volume_surge, 1) if volume_surge > 0 else 1,
                        'rsi': max(min((latest['RSI'] - 70) / 30, 1), 0)
                    }

                    sell_score = (
                        sell_components['trend'] * 0.3 +
                        sell_components['momentum'] * 0.3 +
                        sell_components['volume'] * 0.2 +
                        sell_components['rsi'] * 0.2
                    )

                    hot_coins.append({
                        'pair_id': pair_id,
                        'symbol': pair['symbol'],
                        'name': pair['description'],
                        'interval': interval,
                        'price': current_price,
                        'price_change_24h': price_change,
                        'volume_24h': volume_24h,
                        'volume_surge': volume_surge,
                        'rsi': latest['RSI'],
                        'macd': latest['MACD'],
                        'potential_score': potential_score,
                        'sell_score': sell_score,
                        'signals': signals,
                        'df': df
                    })

                except Exception as e:
                    st.error(f"Error analyzing {pair.get('symbol', 'unknown')}: {str(e)}")
                    continue

            # Clear progress indicators when done
            progress_bar.empty()
            status_text.empty()
            
            hot_coins.sort(key=lambda x: (x['potential_score'], -x['sell_score']), reverse=True)
            return hot_coins

        except Exception as e:
            st.error(f"Error finding hot coins: {str(e)}")
            return []

    def display_analysis(self, coins, top_n=None):
        """Display detailed analysis with Streamlit"""
        if not coins:
            st.warning("No coins found matching criteria")
            return

        interval = coins[0]['interval']
        st.subheader(f"Crypto Analysis ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")

        for i, coin in enumerate(coins[:top_n], 1):
            try:
                with st.expander(f"{i}. {coin['name']} ({coin['symbol'].upper()}) - Score: {coin['potential_score']:.2f}"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**Current Price:** Rp {coin['price']:,.0f}")
                        st.markdown(f"**24h Change:** {coin['price_change_24h']:.2f}%")
                        st.markdown(f"**Volume (24h):** Rp {coin['volume_24h']:,.0f}")
                        st.markdown(f"**Volume Surge:** {coin['volume_surge']:.2f}x")
                        st.markdown(f"**RSI:** {coin['rsi']:.2f}")
                        st.markdown(f"**Score:** {coin['potential_score']:.2f} (Buy) | {coin['sell_score']:.2f} (Sell)")
                    
                    with col2:
                        if coin['signals']:
                            st.markdown("**🔍 SIGNALS:**")
                            for signal in coin['signals']:
                                st.markdown(f"• {signal}")
                    
                    # Get fresh data for detailed analysis
                    df = self.get_klines(coin['symbol'], interval=interval)
                    if df is None or df.empty:
                        st.warning(f"Could not retrieve data for {coin['symbol']}")
                        continue

                    df = self.calculate_indicators(df)
                    price_levels = self.get_price_levels(df, coin['price'])
                    if not price_levels:
                        st.warning(f"Could not calculate price levels for {coin['symbol']}")
                        continue
                        
                    targets = self.calculate_targets(
                        coin['price'],
                        price_levels['atr'],
                        coin['potential_score'],
                        interval
                    )
                    if not targets:
                        st.warning(f"Could not calculate targets for {coin['symbol']}")
                        continue

                    # Display price levels in columns
                    st.markdown("### Price Levels")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**📊 ENTRY LEVELS**")
                        for name, price in price_levels['entry'].items():
                            change = ((price/coin['price'])-1)*100
                            emoji = '🟢' if name == 'aggressive' else '🟡' if name == 'ideal' else '⚪'
                            st.markdown(f"{emoji} **{name.title()}:** Rp {price:,.0f} ({change:+.1f}%)")
                    
                    with col2:
                        st.markdown("**📉 SUPPORT LEVELS**")
                        for name, price in price_levels['support'].items():
                            change = ((price/coin['price'])-1)*100
                            st.markdown(f"💪 **{name.title()}:** Rp {price:,.0f} ({change:+.1f}%)")
                    
                    with col3:
                        st.markdown("**📈 RESISTANCE LEVELS**")
                        for name, price in price_levels['resistance'].items():
                            change = ((price/coin['price'])-1)*100
                            st.markdown(f"🎯 **{name.title()}:** Rp {price:,.0f} ({change:+.1f}%)")
                    
                    # Display trading levels
                    st.markdown("### Trading Levels")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"⛔ **Stop Loss:** Rp {targets['stop_loss']:,.0f} ({((targets['stop_loss']/coin['price'])-1)*100:.1f}%)")
                        for i, target in enumerate(targets['targets'], 1):
                            st.markdown(f"🎯 **Target {i}:** Rp {target:,.0f} ({((target/coin['price'])-1)*100:+.1f}%)")
                    
                    with col2:
                        st.markdown("**📋 TRADING PLAN:**")
                        if coin['potential_score'] > 0.7 and coin['sell_score'] < 0.3:
                            st.markdown("🟢 **Strong Buy Setup**")
                            if interval in ['1', '15', '30']:
                                st.markdown("**Scalping Strategy:**")
                                st.markdown("• Enter: Full position at support")
                                st.markdown("• Stop: Tight stop below entry")
                                st.markdown("• Exit: Quick profit at Target 1")
                            elif interval in ['60', '240']:
                                st.markdown("**Intraday Strategy:**")
                                st.markdown("• Enter: 60% market, 40% support")
                                st.markdown("• Stop: Below key support")
                                st.markdown("• Exit: Scale out at targets")
                            else:
                                st.markdown("**Swing Strategy:**")
                                st.markdown("• Enter: Build position gradually")
                                st.markdown("• Stop: Use wider stops")
                                st.markdown("• Exit: Hold for higher targets")
                        elif coin['sell_score'] > 0.7:
                            st.markdown("🔴 **Sell/Take Profit Setup**")
                            st.markdown("• Set tight stops at recent highs")
                            st.markdown("• Consider scaling out")
                        else:
                            st.markdown("🟡 **Monitor Setup**")
                            st.markdown("• Wait for better entry")
                            st.markdown("• Confirm with volume")
                    
                    st.markdown(f"**Volatility:** {price_levels['volatility']:.1f}%")
                    st.markdown(f"**Risk:Reward:** 1:{targets['risk_reward']:.1f}")
                    
                    # Technical Analysis Chart
                    st.markdown("### Technical Analysis")
                    fig = self.plot_technical_analysis_plotly(coin['symbol'], df, price_levels, targets)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Correlation Analysis
                    st.markdown("### Bitcoin Correlation")
                    corr_fig = self.plot_correlation_plotly(coin['symbol'], interval)
                    if corr_fig:
                        comparison_df = self.get_bitcoin_comparison(coin['symbol'], interval)
                        if comparison_df is not None:
                            correlation = comparison_df['close_alt'].corr(comparison_df['close_btc'])
                            st.markdown(f"**Bitcoin Correlation:** {correlation:.3f}")
                        st.plotly_chart(corr_fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error displaying analysis for {coin['symbol']}: {str(e)}")
                continue

# Streamlit app
def main():
    st.set_page_config(
        page_title="Indodax Hot Finder",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("Indodax Hot Finder 📈")
    st.markdown("Find hot trading opportunities on Indodax cryptocurrency exchange")
    
    finder = IndodaxHotFinder()
    
    # Sidebar for settings
    st.sidebar.header("Settings")
    
    interval_options = {
        '1': '1 Minute',
        '5': '5 Minutes',
        '15': '15 Minutes',
        '30': '30 Minutes',
        '60': '1 Hour',
        '240': '4 Hours',
        '1D': '1 Day',
        '1W': '1 Week'
    }
    
    interval = st.sidebar.selectbox(
        "Timeframe",
        options=list(interval_options.keys()),
        format_func=lambda x: interval_options[x],
        index=4  # Default to 1 hour
    )
    
    min_volume = st.sidebar.slider(
        "Minimum 24h Volume (IDR)",
        min_value=100000000,
        max_value=10000000000,
        value=1000000000,
        step=100000000,
        format="%d"
    )
    
    top_n = st.sidebar.slider(
        "Number of coins to display",
        min_value=5,
        max_value=50,
        value=10
    )
    
    # Single coin analysis
    st.sidebar.header("Single Coin Analysis")
    available_pairs = finder.get_pairs()
    if available_pairs:
        pair_symbols = [pair['symbol'] for pair in available_pairs]
        selected_symbol = st.sidebar.selectbox(
            "Select a coin",
            options=pair_symbols,
            index=0 if 'btc_idr' in pair_symbols else 0
        )
        
        if st.sidebar.button("Analyze Single Coin"):
            with st.spinner(f"Analyzing {selected_symbol}..."):
                df = finder.get_klines(selected_symbol, interval=interval)
                if df is not None and not df.empty:
                    df = finder.calculate_indicators(df)
                    
                    # Get current price
                    tickers = finder.get_ticker_all()
                    ticker_id = next((pair['ticker_id'] for pair in available_pairs if pair['symbol'] == selected_symbol), None)
                    
                    if ticker_id and ticker_id in tickers:
                        current_price = float(tickers[ticker_id]['last'])
                        price_levels = finder.get_price_levels(df, current_price)
                        
                        if price_levels:
                            # Calculate potential score
                            latest = df.iloc[-1]
                            recent_volume = df['volume'].iloc[-1]
                            avg_volume = df['volume'].mean()
                            volume_surge = recent_volume / avg_volume if avg_volume > 0 else 1
                            
                            score_components = {
                                'trend': 1 if latest['MA7'] > latest['MA25'] else 0,
                                'momentum': 1 if latest['MACD'] > latest['Signal'] else 0,
                                'volume': min(volume_surge / 2, 1),
                                'rsi': 1 - abs(50 - latest['RSI']) / 50
                            }
                            
                            potential_score = (
                                score_components['trend'] * 0.3 +
                                score_components['momentum'] * 0.3 +
                                score_components['volume'] * 0.2 +
                                score_components['rsi'] * 0.2
                            )
                            
                            targets = finder.calculate_targets(
                                current_price,
                                price_levels['atr'],
                                potential_score,
                                interval
                            )
                            
                            # Display analysis
                            st.subheader(f"{selected_symbol.upper()} Analysis")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"**Current Price:** Rp {current_price:,.0f}")
                                st.markdown(f"**RSI:** {latest['RSI']:.2f}")
                                st.markdown(f"**MACD:** {latest['MACD']:.6f}")
                                st.markdown(f"**Signal:** {latest['Signal']:.6f}")
                                st.markdown(f"**Potential Score:** {potential_score:.2f}")
                            
                            with col2:
                                st.markdown("**Price Levels:**")
                                st.markdown(f"• Support: Rp {price_levels['support']['moderate']:,.0f}")
                                st.markdown(f"• Resistance: Rp {price_levels['resistance']['first']:,.0f}")
                                st.markdown(f"• Stop Loss: Rp {targets['stop_loss']:,.0f}")
                                st.markdown(f"• Target 1: Rp {targets['targets'][0]:,.0f}")
                                st.markdown(f"• Target 2: Rp {targets['targets'][1]:,.0f}")
                            
                            # Technical Analysis Chart
                            fig = finder.plot_technical_analysis_plotly(selected_symbol, df, price_levels, targets)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Correlation Analysis
                            corr_fig = finder.plot_correlation_plotly(selected_symbol, interval)
                            if corr_fig:
                                st.plotly_chart(corr_fig, use_container_width=True)
                        else:
                            st.error(f"Could not calculate price levels for {selected_symbol}")
                    else:
                        st.error(f"Could not get current price for {selected_symbol}")
                else:
                    st.error(f"Could not retrieve data for {selected_symbol}")
    
    # Main content - Hot coins finder
    if st.button("Find Hot Coins"):
        with st.spinner(f"Scanning market on {interval_options[interval]} timeframe..."):
            hot_coins = finder.find_hot_coins(interval=interval, min_volume_idr=min_volume)
            
            if hot_coins:
                st.success(f"Found {len(hot_coins)} coins with potential")
                finder.display_analysis(hot_coins, top_n=top_n)
            else:
                st.warning("No hot coins found matching your criteria")

if __name__ == "__main__":
    main()
