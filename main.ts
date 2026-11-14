// polymarket-faa-attribution-02 entry (Node 22, standard library only).
//
// Forecast: a random-walk quantile forecast in the shape of the house
// baseline's, with the spread fitted to the market instead of one fixed
// sigma for every market:
//
//   q_tau(h) = clamp(x + z_tau * sigma * s_tau * h^a_tau, 0, 1)
//
// where x is the market's Yes price at the checkpoint, h = 1..240 is the
// minute after it, z_tau is the standard normal's quantile at each level
// (0.05, 0.25, 0.5, 0.75, 0.95), and:
//
//   sigma = clamp(0.018 * x, 0.0001, 0.004)
//     -- the spread is proportional to the price level: a market at 0.30
//        moves in much bigger absolute steps than one at 0.02, and a
//        single fixed sigma fits neither.
//   s_tau = 1.3 for the upper quantiles (z > 0) and 1.15 for the lower
//        ones (z < 0), but only when x < 0.10
//     -- low-priced markets are skewed: bounded below by 0, they can jump
//        up several-fold on news, and they can also collapse toward 0
//        (the FAA rules themselves record 0.49 -> 0.275 and 0.215 -> 0.02
//        one-minute moves). Both tails need extra width; the upside more.
//   a_tau = 0.6 for the outer quantiles (|z| = 1.644854) when x < 0.10,
//        else 0.5
//     -- jump risk grows with the horizon: over 4 hours a low-priced
//        market is more likely to have suffered a jump than over 4
//        minutes, so the 5%/95% band widens faster than sqrt(h). Applied
//        to all markets it hurts (high-priced markets rarely jump), so it
//        is restricted to x < 0.10.
//
// What was measured on the practice (local scorer, ground truth re-fetched
// from Polymarket's public CLOB price history; the scorer reproduces the
// house baseline's published practice numbers exactly, loss 0.363589):
//   - Fixed sigma: best near 0.0008 (0.326921); the baseline's 0.0025
//     over-covers (90% coverage 0.9496).
//   - Price-proportional sigma alone: 0.286251 (c = 0.02, upper skew 1.3
//     for x < 0.05), beating the baseline on all six practice cases.
//   - Realised volatility does NOT help: fitting sigma from each market's
//     trailing 1-minute moves scores 0.296-0.353, and blending it in
//     scores 0.309-0.353 -- the estimate is dominated by rare jumps and
//     by secular trends, and a market that fell hard over the last day is
//     usually quiet over the next 4 hours. A drift term hurts for the
//     same reason (recent trends usually retrace). Lognormal quantiles
//     (0.296) and sigma ~ sqrt(x(1-x)) (0.294) are both worse than the
//     additive price-proportional shape. So this entry uses no history
//     beyond the checkpoint price: it is robust to regime change.
//   - The low-price refinements above (down 1.15, threshold 0.10, outer
//     tail h^0.6 for x < 0.10) improve every slice at once -- overall,
//     markets with x < 0.05, and markets with x < 0.10 -- which is the
//     final's regime: the four FAA markets are low-priced, news-driven
//     attribution questions.
//   - The loss surface is flat near the chosen constants (c 0.016-0.022,
//     up 1.3-1.5, tail exponent 0.55-0.65 all score within ~0.002), so
//     the formula is a structural fit, not a sharp optimum tuned to one
//     window. Practice score of this entry: 0.280981 (median error and
//     90% coverage measured by the local scorer), every case valid.
//     The practice is the NVIDIA event, not the FAA family, so its score
//     is no guide to the final -- the shape, not the number, is what
//     transfers.
//
// Output: {"forecasts":[{"id","q"}]}, q = 240 rows of 5 non-decreasing
// numbers in [0,1], 6 decimals. Deterministic: no randomness, no clock,
// no network; two runs on the same case write identical bytes.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z = [-1.644854, -0.67449, 0, 0.67449, 1.644854];
/** Spread per unit of price level. */
const SIGMA_PER_PRICE = 0.018;
/** Sigma floor / cap (price units per sqrt-minute). */
const SIGMA_FLOOR = 0.0001;
const SIGMA_CAP = 0.004;
/** Price below which a market counts as low-priced. */
const LOW_PRICE = 0.1;
/** Tail multipliers for low-priced markets: upside / downside. */
const UPPER_SKEW = 1.3;
const LOWER_SKEW = 1.15;
/** Horizon exponent of the outer quantiles for low-priced markets. */
const TAIL_EXPONENT = 0.6;

interface CaseMarket {
  id: string;
  prices: number[];
}

interface Case {
  horizon_minutes: number;
  quantiles: number[];
  markets: CaseMarket[];
}

const price6 = (x: number): number => Math.round(Math.min(1, Math.max(0, x)) * 1e6) / 1e6 + 0;

function forecast(m: CaseMarket, horizon: number): number[][] {
  if (m.prices.length === 0) throw new Error(`market ${m.id} has no prices`);
  const x = m.prices[m.prices.length - 1] as number;
  const sigma = Math.min(SIGMA_CAP, Math.max(SIGMA_FLOOR, SIGMA_PER_PRICE * x));
  const low = x < LOW_PRICE;
  // Per-market constants, hoisted out of the per-row loop: the spread and
  // the tail multipliers depend only on the checkpoint price, and the two
  // horizon factors only on h, so each row is five cheap multiplies.
  const up = low ? UPPER_SKEW : 1;
  const down = low ? LOWER_SKEW : 1;
  const rows: number[][] = [];
  for (let h = 1; h <= horizon; h++) {
    const rootH = Math.sqrt(h);
    const tailH = low ? Math.pow(h, TAIL_EXPONENT) : rootH;
    const row = Z.map((z) => {
      const mult = z === 0 ? 1 : z > 0 ? up : down;
      const r = low && Math.abs(z) > 1 ? tailH : rootH;
      return price6(x + z * mult * sigma * r);
    });
    row.sort((a, b) => a - b);
    rows.push(row);
  }
  return rows;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.quantiles.length !== QUANTILES.length || c.quantiles.some((t, j) => t !== QUANTILES[j])) throw new Error(`${file} asks for quantiles ${JSON.stringify(c.quantiles)}, not ${JSON.stringify(QUANTILES)}`);
  const forecasts = c.markets.map((m) => ({ id: m.id, q: forecast(m, c.horizon_minutes) }));
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ forecasts })}\n`);
}
