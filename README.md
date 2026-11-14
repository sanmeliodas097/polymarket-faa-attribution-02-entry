# Polymarket FAA outage attribution forecast

Forecasts the Yes price of every market of Polymarket's FAA outage
attribution family (Iran / foreign actor / hack / AI sandbox escape) at
every minute of the 4 hours after each checkpoint, as five quantiles
(5%, 25%, 50%, 75%, 95%), for the lowest mean pinball loss.

## Method

A random-walk quantile forecast around the checkpoint price:

```
q_tau(h) = clamp(x + z_tau * sigma * s_tau * h^a_tau, 0, 1)
sigma    = clamp(0.018 * x, 0.0001, 0.004)
```

- The spread is proportional to the price level: a market at 0.30 moves in
  much bigger absolute steps than one at 0.02.
- For low-priced markets (x < 0.10) the tails are widened: upside x1.3,
  downside x1.15 — low prices are skewed (bounded below by 0, can jump up
  several-fold on news, and can collapse toward 0).
- For low-priced markets the outer (5%/95%) quantiles widen with h^0.6
  instead of sqrt(h): jump risk grows with the horizon.
- No drift, no realised-volatility fitting: both measured worse on the
  practice (recent trends retrace; vol estimates are dominated by rare
  jumps). The entry uses no history beyond the checkpoint price.

## Practice score (local scorer, ground truth re-fetched from Polymarket)

- House baseline: **0.363589**
- This entry: **0.280981** (−22.7% vs baseline), every case valid,
  beats the baseline on all six practice cases.

The practice is the NVIDIA CEO event, not the FAA family, so the score
is no guide to the final — the shape is what transfers.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Node 22, standard library only. Deterministic: no randomness, no clock,
no network.
