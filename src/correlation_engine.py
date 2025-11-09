"""Utility functions to compute correlation statistics for a configured asset."""

import os
from typing import Dict, Any

import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    sns = None

import config
from src.config_manager import (
    get_setting,
    get_sanitized_symbol,
    sanitize_symbol,
)


def _load_close_series(symbol: str, timeframe: str = "1Day") -> pd.Series:
    """Load a close-price series for the sanitized symbol and timeframe."""

    path = config.get_raw_data_path(symbol, timeframe)
    if not os.path.exists(path):
        return pd.Series(dtype=float)

    df = pd.read_csv(path)
    if df.empty or 'timestamp' not in df.columns or 'close' not in df.columns:
        return pd.Series(dtype=float)

    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp', 'close'])
    if df.empty:
        return pd.Series(dtype=float)

    series = pd.Series(df['close'].values, index=pd.to_datetime(df['timestamp'], utc=True))
    series = series.sort_index()
    series = series[~series.index.duplicated(keep='first')]
    return series.astype(float)


def _align_price_series(series_map: Dict[str, pd.Series]) -> pd.DataFrame:
    """Align multiple price series on their intersection of timestamps."""

    if not series_map:
        return pd.DataFrame()

    combined = pd.concat(series_map, axis=1)
    combined = combined.dropna(how='any')
    return combined


def _ensure_processed_dir() -> None:
    os.makedirs(config.PROCESSED_DATA_PATH, exist_ok=True)


def _processed_path(sanitized: str, stem: str, extension: str) -> str:
    filename = f"{stem}_{sanitized}{extension}"
    return os.path.join(config.PROCESSED_DATA_PATH, filename)


def _plot_heatmap(corr_matrix: pd.DataFrame, output_path: str) -> None:
    if sns is None:
        plt.figure(figsize=(6, 4))
        plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(label='Correlation')
        plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45, ha='right')
        plt.yticks(range(len(corr_matrix.index)), corr_matrix.index)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        return

    plt.figure(figsize=(6, 4))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_rolling(primary_returns: pd.Series,
                  peer_returns: Dict[str, pd.Series],
                  window: int,
                  output_path: str) -> None:
    if not peer_returns:
        return

    plt.figure(figsize=(8, 4))
    for label, series in peer_returns.items():
        aligned = pd.concat([primary_returns, series], axis=1, join='inner')
        if aligned.empty:
            continue
        corr = aligned.iloc[:, 0].rolling(window).corr(aligned.iloc[:, 1])
        corr.name = label
        plt.plot(corr.index, corr.values, label=label)

    plt.title(f'{window}-Day Rolling Correlation vs Primary Asset')
    plt.xlabel('Date')
    plt.ylabel('Correlation')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def compute_asset_correlations(asset_config: Dict[str, Any],
                               window: int = 60) -> Dict[str, str]:
    """Compute correlation artefacts for the configured asset."""

    correlation_assets = get_setting(asset_config, 'correlation.assets', []) or []
    if not correlation_assets:
        print("No correlation assets configured. Skipping correlation analysis.")
        return {}

    primary_symbol = get_setting(asset_config, 'asset.symbol')
    primary_name = get_setting(asset_config, 'asset.name', primary_symbol)
    sanitized_primary = get_sanitized_symbol(asset_config)

    primary_series = _load_close_series(sanitized_primary)
    if primary_series.empty:
        print(f"⚠ Unable to load primary asset data for correlations ({primary_symbol}).")
        return {}

    series_map: Dict[str, pd.Series] = {primary_name: primary_series}
    peer_returns: Dict[str, pd.Series] = {}

    for asset in correlation_assets:
        peer_symbol = asset.get('symbol')
        if not peer_symbol:
            continue
        peer_label = asset.get('name', peer_symbol)
        sanitized_peer = sanitize_symbol(peer_symbol)
        peer_series = _load_close_series(sanitized_peer)
        if peer_series.empty:
            print(f"⚠ No data for correlation asset {peer_symbol}. Skipping.")
            continue
        series_map[peer_label] = peer_series
        peer_returns[peer_label] = peer_series.pct_change().dropna()

    aligned_prices = _align_price_series(series_map)
    if aligned_prices.empty or len(aligned_prices.columns) < 2:
        print("⚠ Insufficient aligned data for correlation analysis.")
        return {}

    returns = aligned_prices.pct_change().dropna()
    corr_matrix = returns.corr()

    _ensure_processed_dir()
    outputs: Dict[str, str] = {}

    matrix_path = _processed_path(sanitized_primary, 'correlation_matrix', '.csv')
    corr_matrix.to_csv(matrix_path)
    outputs['matrix'] = matrix_path
    print(f"✓ Saved correlation matrix to {matrix_path}")

    rolling_output = _processed_path(sanitized_primary, 'rolling_correlations', '.csv')
    returns.to_csv(rolling_output)
    outputs['returns'] = rolling_output
    print(f"✓ Saved return series for rolling correlations to {rolling_output}")

    heatmap_path = _processed_path(sanitized_primary, 'correlation_heatmap', '.png')
    _plot_heatmap(corr_matrix, heatmap_path)
    outputs['heatmap'] = heatmap_path

    primary_returns = returns.iloc[:, 0]
    rolling_plot_path = _processed_path(sanitized_primary, 'rolling_correlations_plot', '.png')
    _plot_rolling(primary_returns, {k: returns[k] for k in returns.columns[1:]}, window, rolling_plot_path)
    outputs['rolling_plot'] = rolling_plot_path

    return outputs


