# 19_apf_gavin

**Question:** What do the pursuers catch against the APF evader, which no policy met in training, and how does that stand beside Table II of Gavin, Lacroix & Bronz 2026?
**Differs from** `18_ent_const_lr`: no training. The policies of `17_const_lr_netpass` and `18_ent_const_lr` fly against the APF evader.
**Runs:** eval only, 2026-11-11, `--seat evader apf`, 1024 episodes. Our FRPN against APF is `--seat pursuer frpn --seat evader apf`.

- The rows against the paths evader are the 4096 episodes of `18_ent_const_lr`. Gavin's rows are his Table II.
- His pursuer deaths are his pursuer crashes plus his double crashes. His `coll_pp` ends the episode and costs 10, as ours, and only a body hit counts (§IV-C).
- His `noSP` pursuers train against the fixed APF evader, with no pool (§V-B).

## Results

| pursuers | evader | catch | pursuer deaths | evader crash | evader free at the end |
|---|---|---|---|---|---|
| his FRPN | APF | 0.550 | 0.394 | 0.055 | 0.001 |
| his `noSP` | APF | 0.652 | 0.140 | 0.151 | 0.057 |
| his PFSP-CTBR | APF | 0.881 | 0.069 | 0.049 | 0.002 |
| our FRPN | `paths` | 0.524 | 0.476 | 0.000 | 0.000 |
| `be848f` | `paths` | 0.498 | 0.433 | 0.000 | 0.069 |
| `9eb432`, last | `paths` | 0.837 | 0.163 | 0.000 | 0.001 |
| `9eb432`, update 850 | `paths` | 0.910 | 0.090 | 0.000 | 0.000 |
| our FRPN | APF | 0.279 | 0.721 | 0.000 | 0.000 |
| `be848f` | APF | 0.454 | 0.393 | 0.000 | 0.153 |
| `9eb432`, last | APF | 0.719 | 0.276 | 0.002 | 0.003 |
| `9eb432`, update 850 | APF | 0.856 | 0.143 | 0.001 | 0.000 |

- Against APF, `9eb432` catches 0.719, and 0.856 at update 850. `be848f` catches 0.454, and our FRPN 0.279.
- Against the paths evader, `9eb432` dies in 0.163 of the episodes, and in 0.090 at update 850. Gavin's learned pursuers die in 0.069–0.140.
- Our FRPN dies in 0.721 of the episodes against our APF. His FRPN dies in 0.394 against his.

His training beside ours. His paper on three pursuers gives no PPO settings, so the lr, the entropy weight and the update come from Gavin & Bronz 2026, Table II:

| | Gavin | `be848f` |
|---|---|---|
| lr | 5e-4, constant | 5e-4, constant |
| action distribution | Gaussian, then a tanh squash | Gaussian, then a clip |
| entropy weight | 0.01 | 0.001 |
| update | 15 epochs × 1 batch, 128-step rollouts | 4 epochs × 32 minibatches, 64-step rollouts |
| activation | ReLU | tanh |
| steps | 5e9 for each side | 5e8 |
| the evader | learns, in a pool; it stays in a 12 × 12 × 6 m box in the centre | `paths`, fixed |
| reward | Table I | the progress reward; before it, his Table I term by term |
