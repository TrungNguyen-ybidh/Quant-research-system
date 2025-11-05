# Week 2 Analysis Summary
**Symbol:** XAU/USD

---

## Volume Distribution Across Timeframes

| Timeframe | Mean Volume | Median Volume |
|-----------|-------------|---------------|
| 1Day | 254,852 | 172,072 |
| 4Hour | 42,485 | 26,766 |
| 1Hour | 11,117 | 6,530 |
| 5Min | 927 | 517 |
| 1Min | 186 | 100 |

**Highest Volume Timeframe:** 1Day (Mean: 254,852)

## Intraday Volume Patterns

### Top 3 Busiest Hours (UTC)

1. **Hour 13:00** - Mean Volume: 26,054
2. **Hour 14:00** - Mean Volume: 25,928
3. **Hour 15:00** - Mean Volume: 19,242

### Quietest 3 Hours (UTC)

1. **Hour 21:00** - Mean Volume: 2,726
2. **Hour 22:00** - Mean Volume: 3,464
3. **Hour 23:00** - Mean Volume: 3,791

## Volume-Price Relationship

**Correlation (Volume vs |Returns|):** 0.4261

**Correlation (Volume vs Returns):** -0.0413

## Volatility Across Timeframes

| Timeframe | Mean ATR | Median ATR | Normalized ATR % |
|-----------|----------|------------|------------------|
| 1Day | 33.26 | 29.28 | 1.400% |
| 4Hour | 12.79 | 10.71 | 0.532% |
| 1Hour | 6.32 | 5.29 | 0.262% |
| 5Min | 1.73 | 1.38 | 0.072% |
| 1Min | 0.73 | 0.56 | 0.030% |

## Intraday Volatility Patterns

Peak volatility hours show concentrated trading activity during specific times of day. The following hours exhibit the highest average ATR values:

### Peak Volatility Hours (UTC)

1. **Hour 15:00** - Mean ATR: 7.02
2. **Hour 16:00** - Mean ATR: 7.00
3. **Hour 17:00** - Mean ATR: 6.92


## Volatility Clustering

**ATR Autocorrelation (lag=1):** 0.9965
**Squared Returns Autocorrelation (lag=1):** 0.1520

**Persistence %:**
- P(High → High): 96.64%
- P(Low → Low): 96.64%

**Regime Durations:**
- Average High Volatility Run Length: 29.8 periods
- Average Low Volatility Run Length: 29.8 periods

---

## Key Findings

- **1Day** timeframe exhibits the highest mean trading volume (254,852) across all timeframes.
- Peak trading activity occurs at **13:00 UTC**, with significantly higher volume compared to quiet hours.
- Strong positive correlation (0.426) between volume and absolute returns indicates that high volume periods predict larger price movements.
- **1Day** timeframe shows the highest normalized ATR (1.400%), while **1Min** exhibits the lowest volatility (0.030%).
- Strong volatility clustering detected (ATR autocorrelation: 0.997), with high-volatility regimes persisting for an average of 29.8 periods.

---