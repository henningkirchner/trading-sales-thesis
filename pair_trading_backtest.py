"""
HOW TO USE THIS SCRIPT
======================

1. INPUT FILES:
   Place the following CSV files in the same directory as this script:
   - HistoricalData_V.csv (Visa OHLC data)
   - HistoricalData_MA.csv (Mastercard OHLC data)
   
   CSV format must include columns: Date, Close/Last, Open, High, Low, Volume
   (Other columns will be ignored)

2. CONFIGURATION:
   Edit the CONFIG SECTION at the top of this file to customize:
   - CSV_VISA_PATH: Path to Visa CSV
   - CSV_MASTERCARD_PATH: Path to Mastercard CSV
   - OUTPUT_DIR: Where to save results (default: 'outputs')
   - START_DATE, END_DATE: Backtest period
   - ROLLING_WINDOWS: List of rolling window sizes (default: [126, 252])
   - ZSCORE_ENTRY: Entry threshold for z-score (default: 2.0)
   - TRANSACTION_COST_BPS: Transaction costs in basis points (default: 10)

3. RUNNING THE SCRIPT:
   From terminal:
   $ python pair_trading_backtest.py
   
   The script will:
   - Load and validate data
   - Run backtests for both window sizes
   - Calculate performance metrics
   - Generate visualizations
   - Save all outputs to the outputs/ folder

4. OUTPUT FILES:
   
   CSV Files:
   - performance_summary.csv: Key metrics for each window variant
   - trade_log_126.csv / trade_log_252.csv: Trade-level details
   - daily_results_126.csv / daily_results_252.csv: Complete daily breakdown
   
   PNG Charts:
   - 01_equity_curves.png: Cumulative returns comparison
   - 02_drawdowns.png: Drawdown analysis
   - 03_spread_zscore_126d.png: Spread and Z-score for 126-day window
   - 04_spread_zscore_252d.png: Spread and Z-score for 252-day window
   - 05_monthly_returns_126d.png: Monthly returns heatmap (126-day)
   - 06_monthly_returns_252d.png: Monthly returns heatmap (252-day)
   - 07_rolling_beta.png: Hedge ratio evolution
   - 08_trade_returns_histogram.png: Trade return distribution
   - 09_performance_comparison.png: Key metrics bar chart
   - 10_monthly_returns.png: Alternative monthly heatmap view

5. KEY CONCEPTS:
   
   Log Spread Model:
   - Reduces scale issues relative to raw price spreads
   - spread = log(Visa) - beta * log(Mastercard)
   - beta = Cov(log_V, log_MA) / Var(log_MA)
   
   Trading Signal:
   - Z-score calculated as: (spread - rolling_mean) / rolling_std
   - Entry: z-score crosses ±2.0
   - Exit: z-score crosses 0
   
   Execution Timing (No Look-Ahead Bias):
   - Signal generated at Close on day t
   - Trade executed at Open on day t+1
   - Returns calculated using next day's open prices
   
   Transaction Costs:
   - Charged per leg (long Visa, short Mastercard = 2 legs)
   - Cost applied whenever position changes
   - Specified in basis points (bps): cost = bps / 10000
   - Total cost = |position_change| * 2 * cost_per_bps

6. INTERPRETING RESULTS:
   
   - Annualized Return: Expected annual return if pattern continues
   - Sharpe Ratio: Return adjusted for volatility (higher = better)
   - Max Drawdown: Largest peak-to-trough decline (risk indicator)
   - Win Rate: Percentage of profitable trades
   - Calmar Ratio: Annual return / max drawdown (risk-adjusted return)
   - Sortino Ratio: Like Sharpe but only counts downside volatility
   
   Compare 126-day vs 252-day:
   - 126-day window: More responsive to recent data, higher turnover
   - 252-day window: More stable, smoother positions, fewer trades
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG SECTION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent

CSV_VISA_PATH = BASE_DIR / 'HistoricalData_V.csv'
CSV_MASTERCARD_PATH = BASE_DIR / 'HistoricalData_MA.csv'

OUTPUT_DIR = BASE_DIR / 'outputs'

HISTORICAL_START_DATE = '2019-01-01'  # Load from 2019 to initialize rolling windows

START_DATE = '2020-01-01'  # Backtest trades begin here (after rolling windows initialized)
END_DATE = '2025-12-31'

# Strategy parameters
ROLLING_WINDOWS = [126, 252]
ZSCORE_ENTRY = 2.0  
ZSCORE_EXIT = 0.0
TRANSACTION_COST_BPS = 10

RISK_FREE_RATE = 0.0

TRADING_DAYS_PER_YEAR = 252


# ============================================================================
# DATA LOADING AND CLEANING
# ============================================================================

def load_and_clean_csv(filepath, asset_name):
    df = pd.read_csv(filepath)
    
    # Ensure required columns exist
    required_cols = ['Date', 'Close/Last', 'Open']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {filepath}")
    
    # Parse date
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    
    # Remove $ sign and convert to float
    df['Close'] = df['Close/Last'].str.replace('$', '').astype(float)
    df['Open'] = df['Open'].str.replace('$', '').astype(float)
    
    # Keep only needed columns
    df = df[['Date', 'Close', 'Open']].copy()
    
    # Rename columns with asset suffix
    df.rename(columns={
        'Close': f'Close_{asset_name}',
        'Open': f'Open_{asset_name}'
    }, inplace=True)
    
    # Sort ascending by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    return df


def merge_data(df_v, df_ma):
    df = pd.merge(df_v, df_ma, on='Date', how='inner')
    
    if df.empty:
        raise ValueError("Merged dataframe is empty. Check file contents.")
    
    return df


def filter_date_range(df, start_date, end_date):
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    df = df[mask].copy().reset_index(drop=True)
    return df


def load_and_prepare_data(visa_path, mastercard_path, start_date, end_date):
    print("Loading data...")
    df_v = load_and_clean_csv(visa_path, 'V')
    df_ma = load_and_clean_csv(mastercard_path, 'MA')
    
    print("Merging data...")
    df = merge_data(df_v, df_ma)
    
    print(f"Filtering date range from {start_date} to {end_date}...")
    df = filter_date_range(df, start_date, end_date)
    
    df = df.dropna().reset_index(drop=True)
    
    print(f"Data loaded: {len(df)} trading days")
    print(f"  Historical + Backtest period: {df['Date'].min().date()} to {df['Date'].max().date()}")
    
    return df


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def add_log_prices(df):
    df['log_price_V'] = np.log(df['Close_V'])
    df['log_price_MA'] = np.log(df['Close_MA'])
    return df


def add_spread(df):
    return df['log_price_V'] - df['log_price_MA']


def add_zscore(spread, window):
    rolling_mean = spread.rolling(window).mean()
    rolling_std = spread.rolling(window).std()
    
    # Avoid division by zero
    zscore = (spread - rolling_mean) / rolling_std.replace(0, np.nan)
    
    return zscore, rolling_mean, rolling_std


def calculate_features(df, window):
    df = df.copy()
    df['spread'] = add_spread(df)
    df['zscore'], df['spread_mean'], df['spread_std'] = add_zscore(df['spread'], window)

    return df


# ============================================================================
# SIGNAL GENERATION
# ============================================================================

def generate_signals(df):
    signal = np.zeros(len(df))

    for i in range(1, len(df)):
        current_z = df['zscore'].iloc[i]
        prev_z = df['zscore'].iloc[i - 1]
        prev_signal = signal[i - 1]

        if np.isnan(current_z):
            signal[i] = prev_signal
            continue

        # Flat -> check entries
        if prev_signal == 0:
            if current_z < -ZSCORE_ENTRY:
                signal[i] = 1
            elif current_z > ZSCORE_ENTRY:
                signal[i] = -1
            else:
                signal[i] = 0

        # In a position -> check exits
        elif prev_signal != 0:
            # Special case: exact zero means true zero-crossing logic
            if ZSCORE_EXIT == 0:
                if prev_signal == 1:
                    # long spread exits when z crosses upward through 0
                    if not np.isnan(prev_z) and prev_z < 0 and current_z >= 0:
                        signal[i] = 0
                    else:
                        signal[i] = 1
                elif prev_signal == -1:
                    # short spread exits when z crosses downward through 0
                    if not np.isnan(prev_z) and prev_z > 0 and current_z <= 0:
                        signal[i] = 0
                    else:
                        signal[i] = -1
            else:
                # Standard symmetric band exit, e.g. 0.5
                if abs(current_z) < ZSCORE_EXIT:
                    signal[i] = 0
                else:
                    signal[i] = prev_signal

    return pd.Series(signal, index=df.index, name='signal')


# ============================================================================
# POSITION AND RETURN CALCULATIONS
# ============================================================================

def calculate_positions_and_returns(df, window, transaction_cost_bps):
    df = df.copy()

    # Signal generated at close t, executed at open t+1
    df['signal'] = df['signal'].shift(1)
    df.loc[df.index[0], 'signal'] = 0

    # Actual position held from today's open to next day's open
    df['position'] = df['signal'].astype(int)

    # Position changes: 1 = enter, -1 = exit, 2 = flip etc.
    df['position_change'] = df['position'].diff().fillna(0)

    # Open-to-open simple returns
    df['ret_V'] = df['Open_V'].shift(-1) / df['Open_V'] - 1
    df['ret_MA'] = df['Open_MA'].shift(-1) / df['Open_MA'] - 1

    # Leg returns, consistent with plain spread
    # position =  1 -> long V, short MA
    # position = -1 -> short V, long MA
    df['asset_return_visa'] = df['position'] * df['ret_V']
    df['asset_return_ma'] = df['position'] * (-df['ret_MA'])

    # Equal capital on both legs: 50% / 50%
    df['strategy_return_gross'] = 0.5 * (
        df['asset_return_visa'] + df['asset_return_ma']
    )

    # 2 legs traded whenever position changes
    # enter/exit of a spread = 2 legs
    df['transaction_cost'] = np.abs(df['position_change']) * (transaction_cost_bps / 10000) * 2

    # Net return
    df['strategy_return_net'] = df['strategy_return_gross'] - df['transaction_cost']

    # Last row has no next-day open
    df = df.iloc[:-1].reset_index(drop=True)

    return df


def calculate_equity_curve(df):
    # Cumulative returns starting from 1.0
    equity = (1 + df['strategy_return_net']).cumprod()
    return equity


def calculate_drawdown(equity):
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max
    return drawdown


# ============================================================================
# TRADE-LEVEL ANALYTICS
# ============================================================================

def generate_trade_log(df):
    trades = []

    entry_idx = None
    entry_date = None
    entry_price_v = None
    entry_price_ma = None
    direction = None

    for i in range(len(df)):
        current_pos = df['position'].iloc[i]
        prev_pos = df['position'].iloc[i-1] if i > 0 else 0

        # Entry
        if current_pos != 0 and prev_pos == 0:
            entry_idx = i
            entry_date = df['Date'].iloc[i]
            entry_price_v = df['Open_V'].iloc[i]
            entry_price_ma = df['Open_MA'].iloc[i]
            direction = 'long_spread' if current_pos == 1 else 'short_spread'

        # Exit
        if current_pos == 0 and prev_pos != 0 and entry_idx is not None:
            exit_idx = i
            exit_date = df['Date'].iloc[i]
            exit_price_v = df['Open_V'].iloc[i]
            exit_price_ma = df['Open_MA'].iloc[i]

            # Price-based gross return, consistent with 50/50 daily PnL logic
            ret_v = exit_price_v / entry_price_v - 1
            ret_ma = exit_price_ma / entry_price_ma - 1

            if direction == 'long_spread':
                trade_return_gross = 0.5 * (ret_v - ret_ma)
            else:
                trade_return_gross = 0.5 * (-ret_v + ret_ma)

            # Round-trip transaction costs:
            # entry = 2 legs, exit = 2 legs
            trade_cost = 4 * (TRANSACTION_COST_BPS / 10000)

            trade_return = trade_return_gross - trade_cost

            duration = (exit_date - entry_date).days

            trades.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'direction': direction,
                'trade_return': trade_return,
                'duration_days': duration
            })

            entry_idx = None
            entry_date = None
            entry_price_v = None
            entry_price_ma = None
            direction = None

    trade_log = pd.DataFrame(trades)

    if len(trade_log) == 0:
        trade_log = pd.DataFrame(
            columns=['entry_date', 'exit_date', 'direction', 'trade_return', 'duration_days']
        )

    return trade_log

# ============================================================================
# PERFORMANCE STATISTICS
# ============================================================================

def calculate_performance_metrics(df, trade_log):
    returns = df['strategy_return_net'].dropna()
    
    if len(returns) == 0:
        return {}
    
    total_return = df['equity_curve'].iloc[-1] - 1
    cum_return = df['equity_curve'].iloc[-1]
    
    num_years = len(df) / TRADING_DAYS_PER_YEAR
    annualized_return = (cum_return ** (1 / num_years)) - 1 if num_years > 0 else 0
    
    daily_volatility = returns.std()
    annualized_volatility = daily_volatility * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    sharpe_ratio = (annualized_return - RISK_FREE_RATE) / annualized_volatility if annualized_volatility > 0 else 0
    
    downside_returns = returns[returns < 0]
    downside_volatility = downside_returns.std()
    annualized_downside_vol = downside_volatility * np.sqrt(TRADING_DAYS_PER_YEAR)
    sortino_ratio = (annualized_return - RISK_FREE_RATE) / annualized_downside_vol if annualized_downside_vol > 0 else 0
    
    max_dd = df['drawdown'].min()
    
    calmar_ratio = annualized_return / abs(max_dd) if max_dd != 0 else 0
    
    # Trade statistics
    num_completed_trades = len(trade_log)
    if num_completed_trades > 0:
        winning_trades = len(trade_log[trade_log['trade_return'] > 0])
        win_rate = winning_trades / num_completed_trades
        avg_trade_return = trade_log['trade_return'].mean()
        median_trade_return = trade_log['trade_return'].median()
        avg_trade_duration = trade_log['duration_days'].mean()
    else:
        win_rate = 0
        avg_trade_return = 0
        median_trade_return = 0
        avg_trade_duration = 0
    
    in_market_days = (df['position'] != 0).sum()
    pct_in_market = in_market_days / len(df) if len(df) > 0 else 0
    
    num_position_changes = (df['position'].diff() != 0).sum()
    
    metrics = {
        'cumulative_return': total_return,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_dd,
        'calmar_ratio': calmar_ratio,
        'num_completed_trades': int(num_completed_trades),
        'win_rate': win_rate,
        'avg_trade_return': avg_trade_return,
        'median_trade_return': median_trade_return,
        'avg_trade_duration_days': avg_trade_duration,
        'pct_days_in_market': pct_in_market,
        'num_position_changes': int(num_position_changes)
    }
    
    return metrics


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_equity_curves(results_dict, output_dir):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for window, data in results_dict.items():
        ax.plot(data['df']['Date'], data['df']['equity_curve'], label=f'{window}-day window', linewidth=2)
    
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Equity Value', fontsize=11)
    ax.set_title('Pair Trading Strategy: Equity Curve Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_equity_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 01_equity_curves.png")


def plot_drawdowns(results_dict, output_dir):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for window, data in results_dict.items():
        ax.fill_between(data['df']['Date'], data['df']['drawdown'], 0, 
                        label=f'{window}-day window', alpha=0.6)
    
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Drawdown', fontsize=11)
    ax.set_title('Pair Trading Strategy: Drawdown Curves', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_drawdowns.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 02_drawdowns.png")


def plot_spread_zscore(df, window_days, output_dir, plot_num, spread_label=None):
    if spread_label is None:
        spread_label = f"{window_days}-day"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot spread with mean
    ax1.plot(df['Date'], df['spread'], label='Spread', linewidth=1, alpha=0.7, color='steelblue')
    ax1.plot(df['Date'], df['spread_mean'], label='Rolling Mean', linewidth=2, color='red', linestyle='--')
    ax1.fill_between(df['Date'], df['spread_mean'] - df['spread_std'], 
                     df['spread_mean'] + df['spread_std'], alpha=0.2, color='gray', label='±1 Std Dev')
    ax1.set_ylabel('Spread (log)', fontsize=11)
    ax1.set_title(f'Log Spread - {spread_label} Window', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    
    # Plot z-score with thresholds
    ax2.plot(df['Date'], df['zscore'], label='Z-Score', linewidth=1.5, color='darkblue')
    ax2.axhline(y=ZSCORE_ENTRY, color='red', linestyle='--', linewidth=1.5, label='Entry Threshold (+2)')
    ax2.axhline(y=-ZSCORE_ENTRY, color='green', linestyle='--', linewidth=1.5, label='Entry Threshold (-2)')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5, label='Exit Level (0)')
    ax2.fill_between(df['Date'], -ZSCORE_ENTRY, ZSCORE_ENTRY, alpha=0.1, color='gray')
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('Z-Score', fontsize=11)
    ax2.set_title(f'Z-Score Trading Signal - {spread_label} Window', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'0{plot_num}_spread_zscore_{window_days}d.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: 0{plot_num}_spread_zscore_{window_days}d.png")


def plot_rolling_betas(results_dict, output_dir):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for window, data in results_dict.items():
        ax.plot(data['df']['Date'], data['df']['beta'], label=f'{window}-day window', linewidth=1.5)
    
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Hedge Ratio (Beta)', fontsize=11)
    ax.set_title('Rolling Hedge Ratio: Visa vs Mastercard', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '07_rolling_beta.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 07_rolling_beta.png")


def plot_spread_with_trades(df, trade_log, window_days, output_dir, plot_num):
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(df['Date'], df['spread'], label='Log Spread', linewidth=1.5, color='steelblue', alpha=0.7)
    ax.plot(df['Date'], df['spread_mean'], label='Rolling Mean', linewidth=1.5, color='orange', linestyle='--', alpha=0.7)
    
    for _, trade in trade_log.iterrows():
        entry_date = pd.to_datetime(trade['entry_date'])
        exit_date = pd.to_datetime(trade['exit_date'])
        direction = trade['direction']
        trade_return = trade['trade_return']
        
        # Get spread values at entry and exit dates
        entry_spread = df[df['Date'] == entry_date]['spread'].values
        exit_spread = df[df['Date'] == exit_date]['spread'].values
        
        if len(entry_spread) > 0 and len(exit_spread) > 0:
            entry_val = entry_spread[0]
            exit_val = exit_spread[0]

            color = 'green' if 'long' in direction else 'red'
            marker_entry = '^' if 'long' in direction else 'v'
            ax.scatter(entry_date, entry_val, marker=marker_entry, s=150, color=color, 
                      edgecolors='black', linewidth=1.5, zorder=5, alpha=0.8)
            
            ax.scatter(exit_date, exit_val, marker='X', s=150, color=color, 
                      edgecolors='black', linewidth=1.5, zorder=5, alpha=0.8)
            
            ax.plot([entry_date, exit_date], [entry_val, exit_val], 
                   color=color, linewidth=1, linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Log Spread', fontsize=11)
    ax.set_title(f'Spread Movement with Trade Markers - {window_days}-Day Window', 
                fontsize=13, fontweight='bold')
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markeredgecolor='black',
               markersize=10, label='Long Entry'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='red', markeredgecolor='black',
               markersize=10, label='Short Entry'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='green', markeredgecolor='black',
               markersize=10, label='Exit (Long)'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='red', markeredgecolor='black',
               markersize=10, label='Exit (Short)'),
        Line2D([0], [0], color='steelblue', linewidth=1.5, label='Spread'),
        Line2D([0], [0], color='orange', linewidth=1.5, linestyle='--', label='Rolling Mean'),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'0{plot_num}_spread_with_trades_{window_days}d.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: 0{plot_num}_spread_with_trades_{window_days}d.png")


def plot_trade_returns_histogram(trade_logs_dict, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    for idx, (window, trade_log) in enumerate(trade_logs_dict.items()):
        ax = axes[idx]
        
        if len(trade_log) > 0:
            ax.hist(trade_log['trade_return'] * 100, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
            ax.axvline(trade_log['trade_return'].mean() * 100, color='red', linestyle='--', 
                      linewidth=2, label=f'Mean: {trade_log["trade_return"].mean()*100:.2f}%')
            ax.set_xlabel('Trade Return (%)', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title(f'{window}-Day Window (n={len(trade_log)} trades)', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'No trades', ha='center', va='center', fontsize=12)
            ax.set_title(f'{window}-Day Window', fontsize=12, fontweight='bold')
    
    fig.suptitle('Distribution of Trade Returns', fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '08_trade_returns_histogram.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 08_trade_returns_histogram.png")


def plot_monthly_returns_heatmap(df, output_dir, window_days):
    df['YearMonth'] = df['Date'].dt.to_period('M')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    
    monthly_returns = df.groupby('YearMonth')['strategy_return_net'].sum() * 100
    monthly_returns_df = pd.DataFrame(monthly_returns)
    monthly_returns_df['Year'] = monthly_returns_df.index.year
    monthly_returns_df['Month'] = monthly_returns_df.index.month
    
    pivot_table = monthly_returns_df.pivot_table(
        values='strategy_return_net', index='Year', columns='Month', aggfunc='sum'
    )
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create heatmap
    im = ax.imshow(pivot_table, cmap='RdYlGn', aspect='auto', vmin=-5, vmax=5)

    ax.set_xticks(range(12))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.set_yticks(range(len(pivot_table)))
    ax.set_yticklabels(pivot_table.index)
    
    ax.set_xlabel('Month', fontsize=11)
    ax.set_ylabel('Year', fontsize=11)
    ax.set_title(f'Monthly Returns Heatmap - {window_days}-Day Window (%)', fontsize=13, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Return (%)', fontsize=11)

    for i in range(len(pivot_table)):
        for j in range(12):
            if not np.isnan(pivot_table.iloc[i, j]):
                text = ax.text(j, i, f'{pivot_table.iloc[i, j]:.1f}',
                             ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'09_monthly_returns_{window_days}d.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: 09_monthly_returns_{window_days}d.png")


def plot_performance_comparison(metrics_dict, output_dir):
    windows = list(metrics_dict.keys())
    
    # Select key metrics for comparison
    metric_names = ['Annualized Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate']
    metric_keys = ['annualized_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
    
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()
    
    for idx, (metric_name, metric_key) in enumerate(zip(metric_names, metric_keys)):
        ax = axes[idx]
        
        values = [metrics_dict[w][metric_key] for w in windows]
        colors = ['steelblue', 'coral']
        
        bars = ax.bar([str(w) for w in windows], values, color=colors, edgecolor='black', alpha=0.7)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            if metric_key == 'max_drawdown':
                label = f'{val*100:.2f}%'
            elif metric_key == 'annualized_return':
                label = f'{val*100:.2f}%'
            elif metric_key == 'win_rate':
                label = f'{val*100:.1f}%'
            else:
                label = f'{val:.2f}'
            
            ax.text(bar.get_x() + bar.get_width()/2, height,
                   label, ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_xlabel('Window Size (days)', fontsize=11)
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Performance Metric Comparison: 126-Day vs 252-Day Window', 
                 fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '10_performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 10_performance_comparison.png")


def generate_all_visuals(results_dict, trade_logs_dict, metrics_dict, output_dir):
    print("\nGenerating visualizations...")
    
    # Create visualization input dict using backtest period data
    # Maintain the nested dict structure expected by plot functions
    results_for_viz = {}
    for window in results_dict.keys():
        results_for_viz[window] = {'df': results_dict[window]['df_backtest']}
    
    plot_equity_curves(results_for_viz, output_dir)
    plot_drawdowns(results_for_viz, output_dir)
    
    for window in [126, 252]:
        plot_num = 3 if window == 126 else 5
        plot_spread_zscore(results_dict[window]['df_backtest'], window, output_dir, plot_num)
    
    for window in [126, 252]:
        plot_num = 4 if window == 126 else 6
        plot_spread_with_trades(results_dict[window]['df_backtest'], trade_logs_dict[window], 
                               window, output_dir, plot_num)
    
    plot_trade_returns_histogram(trade_logs_dict, output_dir)
    
    for window in [126, 252]:
        plot_monthly_returns_heatmap(results_dict[window]['df_backtest'], output_dir, window)
    
    plot_performance_comparison(metrics_dict, output_dir)
    
    print("\nAll visualizations completed.\n")


# ============================================================================
# OUTPUT SAVING
# ============================================================================

def save_performance_summary(metrics_dict, output_dir):
    rows = []
    for window, metrics in metrics_dict.items():
        row = {'window_days': window}
        row.update(metrics)
        rows.append(row)
    
    summary_df = pd.DataFrame(rows)
    summary_path = os.path.join(output_dir, 'performance_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Saved: performance_summary.csv")


def save_trade_logs(trade_logs_dict, output_dir):
    for window, trade_log in trade_logs_dict.items():
        filename = f'trade_log_{window}.csv'
        trade_log.to_csv(os.path.join(output_dir, filename), index=False)
        print(f"✓ Saved: {filename}")


def save_daily_results(results_dict, output_dir):
    columns_to_save = [
    'Date', 'Open_V', 'Close_V', 'Open_MA', 'Close_MA',
    'spread', 'spread_mean', 'spread_std', 'zscore',
    'signal', 'position', 'asset_return_visa', 'asset_return_ma',
    'strategy_return_gross', 'transaction_cost', 'strategy_return_net',
    'equity_curve', 'drawdown'
    ]
    
    for window, data in results_dict.items():
        # Use df_backtest (backtest period only, excluding 2019 historical)
        df = data['df_backtest']
        df_save = df[columns_to_save].copy()
        filename = f'daily_results_{window}.csv'
        df_save.to_csv(os.path.join(output_dir, filename), index=False)
        print(f"✓ Saved: {filename}")


def create_output_directory(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory ready: {output_dir}")


# ============================================================================
# MAIN BACKTEST ENGINE
# ============================================================================

def run_backtest(data, window):
    print(f"\n--- Running backtest for {window}-day rolling window ---")
    df = calculate_features(data, window)
    df['signal'] = generate_signals(df)
    df = calculate_positions_and_returns(df, window, TRANSACTION_COST_BPS)
    df['equity_curve'] = calculate_equity_curve(df)
    df['drawdown'] = calculate_drawdown(df['equity_curve'])
    trade_log = generate_trade_log(df)
    metrics = calculate_performance_metrics(df, trade_log)

    print(f"Cumulative Return: {metrics['cumulative_return']*100:.2f}%")
    print(f"Annualized Return: {metrics['annualized_return']*100:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    print(f"Completed Trades: {metrics['num_completed_trades']}")
    print(f"Win Rate: {metrics['win_rate']*100:.1f}%")
    
    return df, trade_log, metrics


def main():
    
    print("="*70)
    print("PAIR TRADING BACKTEST: VISA vs MASTERCARD")
    print("="*70)
    print(f"Data loading period: {HISTORICAL_START_DATE} to {END_DATE}")
    print(f"  (includes historical buffer for rolling window initialization)")
    print(f"Backtest trading period: {START_DATE} to {END_DATE}")
    
    create_output_directory(OUTPUT_DIR)
    data = load_and_prepare_data(CSV_VISA_PATH, CSV_MASTERCARD_PATH, HISTORICAL_START_DATE, END_DATE)
    data = add_log_prices(data)

    results_dict = {}
    trade_logs_dict = {}
    metrics_dict = {}

    for window in ROLLING_WINDOWS:
        df, trade_log, metrics = run_backtest(data, window)
        
        # Filter to actual backtest period (START_DATE onwards) for reporting
        df_backtest = df[df['Date'] >= START_DATE].copy().reset_index(drop=True)
        
        # Recalculate metrics using only backtest period data
        trade_log_filtered = trade_log[
            (trade_log['entry_date'] >= START_DATE)
        ].copy().reset_index(drop=True)
        metrics_filtered = calculate_performance_metrics(df_backtest, trade_log_filtered)
        
        # Store both full data (for charts) and filtered results (for metrics)
        results_dict[window] = {'df': df, 'df_backtest': df_backtest, 'trade_log': trade_log_filtered}
        trade_logs_dict[window] = trade_log_filtered
        metrics_dict[window] = metrics_filtered
    
    # Generate and save visualizations
    generate_all_visuals(results_dict, trade_logs_dict, metrics_dict, OUTPUT_DIR)
    
    # Save outputs
    print("\nSaving outputs...")
    save_performance_summary(metrics_dict, OUTPUT_DIR)
    save_trade_logs(trade_logs_dict, OUTPUT_DIR)
    save_daily_results(results_dict, OUTPUT_DIR)
    
    print("\n" + "="*70)
    print("BACKTEST COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  - performance_summary.csv")
    print("  - trade_log_126.csv")
    print("  - trade_log_252.csv")
    print("  - daily_results_126.csv")
    print("  - daily_results_252.csv")
    print("  - 10 PNG chart files")
    print("\n" + "="*70)

if __name__ == '__main__':
    main()