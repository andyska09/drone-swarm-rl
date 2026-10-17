# Last-minute avoidance — literature review

The task: two drones fly at each other, each on its own line. Each drone sees the other
only inside a range `d`, through several sensor types, and nothing is sent by radio. They
must dodge at the last moment, then go back to the line. Encounters are random, so a blind
rule such as "always fly a bit to the right" fails.

The main review is [lit_review.md](lit_review.md). The head-on case with full sight
already exists as the `pair` preset in [task_circle_swap.md](task_circle_swap.md).

Each entry says what the paper does, then why it matters here. Lines marked **our
arithmetic** or **our reading** are ours, not the paper's. "Title only" means only the
title, authors and venue were checked, not the content.

---

## 0. Words used below

| word | meaning |
|---|---|
| sensing range `d` | the farthest distance at which a sensor detects the other drone |
| field of view (FOV) | the angle a sensor covers. A forward cone sees nothing behind or far to the side |
| latency `τ` | time from "the other drone enters sensor range" to "my drone starts to move away" |
| closing speed | how fast the gap shrinks. Two drones head-on at 3 m/s each close at 6 m/s |
| closest point of approach (CPA) | the moment two drones are nearest if nobody changes course. The gap at that moment is the **miss distance** |
| time to contact | gap ÷ closing speed. An eye can read it from how fast the image of an object grows ("looming"), with no distance needed — [Lee 1976](https://doi.org/10.1068/p050437) |
| bearing | the direction to the other drone, with no distance |
| range-rate | how fast the distance changes. Radar measures it directly |
| modality | one sensor type: camera, LiDAR, radar, microphone, and others |
| fusion | combining sensors. **Early** fusion merges raw data first. **Late** fusion lets each sensor make its own estimate, then merges the estimates |
| false alarm / miss | the sensor reports a drone that is not there / does not report one that is |
| tracker | a filter (usually a Kalman filter) that turns noisy detections into a smooth position and velocity |
| teacher / student | in simulation a teacher policy sees the exact state. A student policy that sees only the sensors learns to copy it |
| deadlock, "reciprocal dance" | both drones dodge to the same side again and again, or stop in front of each other |
| shortcut | a rule that wins on the training data and fails on new cases — [Geirhos et al. 2020](https://arxiv.org/abs/2004.07780) |

---

## 1. How much warning a dodge needs

- **Falanga, Kim & Scaramuzza 2019, *How Fast Is Too Fast? The Role of Perception Latency
  in High-Speed Sense and Avoid*** — RA-L,
  [10.1109/LRA.2019.2898117](https://doi.org/10.1109/LRA.2019.2898117)
  A robot keeps its forward speed `v` and may only move sideways, with sideways
  acceleration at most `a`. It must move sideways by `r` (its own size plus the
  obstacle's). The fastest move is full push, then full brake, which takes
  `2·sqrt(r/a)`. The obstacle appears at distance `s` (our `d`) and the robot reacts after
  latency `τ`. It survives if:

  ```
  s/v − τ ≥ 2·sqrt(r/a)              (eq. 7)
  max latency  τ̄ = s/v − 2·sqrt(r/a)   (eq. 8)
  max speed    v̄ = s / (τ + 2·sqrt(r/a))   (eq. 9)
  ```

  If the robot need not stop its sideways motion, the move is shorter: `sqrt(2r/a)`.
  Their numbers: monocular camera `s` = 2–6 m with `τ` = 26–40 ms at 50 Hz; stereo `s` =
  2–8 m with `τ` = 17–70 ms; event camera `s` = 2–8 m, above 10 m possible. They test `a`
  = 10, 25, 50 and 200 m/s², about thrust-to-weight 1.5, 2.8, 5.2 and 20. The model
  ignores the time a quadrotor needs to tilt before the sideways push starts.
  **Why:** it turns "see at the last minute" into one number we can set.

- **Mandaokar & Rinner 2026, *RADAR Perception for Dynamic Obstacle Avoidance onboard
  small-scale Quadrotor UAVs*** — [arXiv:2608.01855](https://arxiv.org/abs/2608.01855)
  Derives bounds that tie sensing range, relative speed and control delay together, for
  objects thrown at a small quadrotor, then flies it: 77–81 GHz mmWave radar, a
  multiple-model tracker, and a control barrier function that outputs the evasive
  acceleration. About 14 ms from sensing to command on a Raspberry Pi 4B. 390 throws, in
  light, dark and smoke.
  **Why:** the same bound as Falanga, for a moving object, with a sensor that measures
  speed directly.

**Our arithmetic** for this repo. Two drones head-on, contact at 0.80 m between centres,
`τ` = 0.05 s. Two drones dodging opposite ways is the same as one drone dodging with
twice the acceleration, so `r` = 0.80 m if one dodges and 0.40 m each if both dodge.
From eq. 7:

```
d = closing speed × (τ + 2·sqrt(r/a))
```

| speed of each drone | closing | one dodges, a = 5 | a = 10 | a = 25 | both dodge, a = 5 | a = 10 | a = 25 |
|---|---|---|---|---|---|---|---|
| 2 m/s | 4 m/s | 3.40 m | 2.46 m | 1.63 m | 2.46 m | 1.80 m | 1.21 m |
| 3 m/s | 6 m/s | 5.10 m | 3.69 m | 2.45 m | 3.69 m | 2.70 m | 1.82 m |
| 5 m/s | 10 m/s | 8.50 m | 6.16 m | 4.08 m | 6.16 m | 4.50 m | 3.03 m |

`a` is in m/s². Each extra 0.1 s of latency adds closing speed × 0.1 s: 0.4 m at 4 m/s,
1.0 m at 10 m/s. Our plant (2.0 kg, four motors at most 7800 rpm, `kf` = 2.7087e-7) has
thrust-to-weight about 3.4, so at full thrust it can push about 31 m/s² sideways while
it holds height — but only after a tilt of about 73°. A sideways offset `δ` at the start
cuts the needed move to `0.80 − δ`. At `δ ≥ 0.80 m` no dodge is needed at all.

---

## 2. Sensors that can see another drone

Each number is what that source reports for its own setup. Rows are not comparable.
"—" means no checked number.

| modality | measures | range for a small drone | timing | source |
|---|---|---|---|---|
| RGB camera + neural detector | bearing; distance only from known size | Det-Fly images: target 10–100 m away | frame rate + network | Det-Fly, Schilling 2021 |
| monocular camera (model) | — | 2–6 m | 26–40 ms | Falanga 2019 |
| stereo / depth camera | 3D position | 2–8 m (model); 9.5 m best real detection of a drone | 17–70 ms | Falanga 2019, Carrio 2018 |
| event camera | per-pixel brightness change | 2–8 m (model), above 10 m possible | microseconds per event | Falanga 2019 |
| 3D LiDAR | 3D points | micro UAV at 20 m, almost 100% recall, 0.2 m accuracy | 20 ms delay | VOFOD 2025 |
| 8×8 multi-zone ToF | 64 distances | 0.2–4 m | 15 fps | Müller 2022 |
| mmWave radar | range, speed, angle | — | ~14 ms end to end | Mandaokar 2026 |
| UV LEDs + UV cameras (UVDAR) | bearing, identity | up to 15 m | — | Petráček 2021 |
| microphones | bearing from sound | — | — | Cabrera-Ponce 2020 |
| thermal camera | heat image | as good as visible at lower resolution (ground test) | — | Svanström 2021 |

### 2.1 Cameras

- **Zheng et al. 2021, *Air-to-Air Visual Detection of Micro-UAVs* (Det-Fly)** — RA-L,
  [10.1109/LRA.2021.3056059](https://doi.org/10.1109/LRA.2021.3056059)
  13,271 images of a flying DJI Mavic, taken from a flying DJI M210. Target 10–100 m
  away, altitude 20–110 m, many backgrounds and lights. Eight deep detectors compared.
  **Why:** the camera view from another drone. Note it starts at 10 m; our `d` is shorter.

- **Schilling, Schiano & Floreano 2021, *Vision-Based Drone Flocking in Outdoor
  Environments*** — RA-L, [arXiv:2012.01245](https://arxiv.org/abs/2012.01245)
  Several cameras per drone, so no blind spots. A CNN detects nearby drones, distance
  comes from their apparent size, a multi-agent tracker estimates relative position and
  velocity, a flocking rule flies. No radio, no markers. Three real drones outdoors.
  **Why:** the full camera chain, and a camera model we can simulate: bearing plus
  apparent size.

- **Vrba & Saska 2020, *Marker-Less Micro Aerial Vehicle Detection and Localization
  Using CNNs*** — RA-L, [10.1109/LRA.2020.2972819](https://doi.org/10.1109/LRA.2020.2972819);
  **Vrba, Hert & Saska 2019, *Onboard Marker-Less Detection and Localization of
  Non-Cooperating Drones*** — RA-L, [10.1109/LRA.2019.2927130](https://doi.org/10.1109/LRA.2019.2927130)
  Title only. The MRS group's onboard detectors for other drones.

- **Pliska, Vrba et al. 2026, *Towards Agile Vision-Based Multi-UAV Flight: Revisiting
  State Estimation*** — [arXiv:2609.39611](https://arxiv.org/abs/2609.39611)
  Filters fed only with detected positions lag about 300 ms in acceleration, at every
  agility tested (3–21 m/s²). Adding the neighbour's tilt, seen by the detector, cuts
  velocity error by 40% and acceleration error by 57%.
  **Why:** 300 ms is a large part of the budget in section 1. The other drone's tilt is
  worth observing.

- **Carrio et al. 2018, *Drone Detection Using Depth Maps*** —
  [arXiv:1808.00259](https://arxiv.org/abs/1808.00259)
  Detector trained on 6k synthetic depth maps of drones. Gives full 3D position. Real
  tests with drones at up to 2 m/s: precision 98.7%, recall 74.7%, best range 9.5 m.
  **Why:** a depth camera gives range and bearing in one sensor, but only to about 10 m,
  and it misses one drone in four.

### 2.2 Event cameras

- **Falanga, Kleber & Scaramuzza 2020, *Dynamic obstacle avoidance for quadrotors with
  event cameras*** — Science Robotics,
  [10.1126/scirobotics.aaz9712](https://doi.org/10.1126/scirobotics.aaz9712)
  Title only. A quadrotor dodges objects thrown at it, using event cameras.
- **Sanket et al. 2020, *EVDodgeNet*** — ICRA, [arXiv:1906.02919](https://arxiv.org/abs/1906.02919)
  Shallow networks on one event camera estimate own motion and the obstacles' motion.
  Trained in simulation, no fine-tuning. 70% success in real tests, including unknown
  shapes and low light.
- **Thakur et al. 2026, *Relative State Estimation using Event-Based Propeller Sensing*** —
  [arXiv:2604.18289](https://arxiv.org/abs/2604.18289)
  An event camera tracks the other drone's propellers. It estimates each propeller's
  speed (under 3% error, five real outdoor flights) and the drone's tilt from the
  propeller's ellipse.
  **Why:** propeller speed is thrust, so this sees what the other drone is about to do.
- **Gallego et al., *Event-based Vision: A Survey*** —
  [arXiv:1904.08405](https://arxiv.org/abs/1904.08405). Reference for how the sensor works.

### 2.3 LiDAR and ToF

- **Vrba, Walter, Pritzl, Pliska et al. 2025, *On Onboard LiDAR-based Flying Object
  Detection*** — T-RO, [arXiv:2303.05404](https://arxiv.org/abs/2303.05404)
  MRS group. 3D occupancy voxel map plus a cluster tracker on an Ouster OS1-128. At 20 m
  it detects a micro-scale UAV with almost 100% recall, 0.2 m accuracy and 20 ms delay.
  The tracker removes occasional false alarms.
- **Choi et al. 2026, *Communication-Free Collective Navigation for a Swarm of UAVs via
  LiDAR-Based Deep RL*** — [arXiv:2601.13657](https://arxiv.org/abs/2601.13657)
  One 360° LiDAR per drone. Point clusters and an extended Kalman filter track the
  neighbours. The RL policy (Isaac Sim) gets up to six neighbours, with zero padding and a
  0/1 mask when fewer are seen (occlusion, small vertical FOV), plus an occupancy grid.
  Only the leader knows the goal. Five real drones.
  **Why:** the same `others` + `others_mask` layout as our `Obs`, fed by a real sensor.
- **Müller et al. 2022, *Robust and Efficient Depth-based Obstacle Avoidance for
  Autonomous Miniaturized UAVs*** — [arXiv:2208.12624](https://arxiv.org/abs/2208.12624)
  One 8×8-zone ToF sensor on a 35 g Crazyflie 2.1. Range about 0.2–4 m, 15 fps, 0.3% of
  the onboard compute. 100% reliable at 0.5 m/s indoors.
  **Why:** what a cheap range sensor gives: 64 distances, 4 m.
- **Chiu et al. 2025, *LEARN*** — [arXiv:2511.17765](https://arxiv.org/abs/2511.17765)
  Low-resolution ToF, a simple planner and a small attention-based RL policy. Six
  Crazyflies, fully onboard, up to 2.0 m/s, through 0.2 m gaps.
  **Why:** multi-drone RL on short-range ToF only.

### 2.4 Radar

- **Mandaokar & Rinner 2026** — section 1. They argue that cameras and LiDARs do not
  measure speed directly, so they need extra processing and latency.
- **Arroyo et al. 2023, *Segmentation of Drone Collision Hazards in Airborne RADAR Point
  Clouds Using PointNet*** — [arXiv:2311.03221](https://arxiv.org/abs/2311.03221)
  Radar on an aircraft. Five classes: two drone types, a light aircraft, ground,
  structures. 94% accuracy.

### 2.5 Markers, sound, heat, looming

- **Walter, Staub, Franchi & Saska 2019, *UVDAR System for Visual Relative
  Localization*** — RA-L, [10.1109/LRA.2019.2901683](https://doi.org/10.1109/LRA.2019.2901683)
  Title only. UV LEDs on each drone, UV-sensitive cameras on the others.
- **Petráček, Walter, Báča & Saska 2021, *Bio-inspired compact swarms of UAVs without
  communication and external localization*** — Bioinspir. Biomim.,
  [10.1088/1748-3190/abc6b3](https://doi.org/10.1088/1748-3190/abc6b3)
  Outdoor swarm with UVDAR only, no radio. LEDs blink at 6, 15 and 30 Hz so drones tell
  each other apart. Detection up to 15 m.
  **Why:** the group behind our physics reference already flies no-radio swarms.
- **Cabrera-Ponce, Martinez-Carranza & Rascon 2020, *Detection of nearby UAVs using a
  multi-microphone array on board a UAV*** — IJMAV,
  [10.1177/1756829320925748](https://doi.org/10.1177/1756829320925748). Title only.
  A drone's own propeller noise is a known problem for onboard microphones
  ([Wu et al. 2025](https://arxiv.org/abs/2508.06310), title only).
- **Svanström, Englund & Alonso-Fernandez 2021, *Real-Time Drone Detection and Tracking
  With Visible, Thermal and Acoustic Sensors*** — ICPR, [arXiv:2007.07396](https://arxiv.org/abs/2007.07396)
  Ground system: visible camera, thermal camera, microphone. Thermal works as well as
  visible at lower resolution. Fusion cuts false detections. Performance measured
  against distance.
  **Why:** the clearest "fusion helps against false alarms" result.
- **Yuan et al. 2024, *MMAUD: A Comprehensive Multi-Modal Anti-UAV Dataset*** —
  [arXiv:2402.03706](https://arxiv.org/abs/2402.03706)
  Stereo cameras, several LiDARs, radars and microphone arrays, with precise ground
  truth. Ground-based.
  **Why:** real multi-sensor data to check our noise models against.
- **Zhao et al. 2019, *An LGMD Based Competitive Collision Avoidance Strategy for UAV*** —
  [arXiv:1904.07206](https://arxiv.org/abs/1904.07206)
  Locust looming-detector neurons, one for each quarter of the image. An object looming
  from the left pushes the drone right.
  **Why:** a looming-only sensor with the side choice built in.

---

## 3. Combining sensors

- **Katz et al. 2025, *Aircraft Collision Avoidance Systems: Technological Challenges and
  Solutions*** — [arXiv:2510.20916](https://arxiv.org/abs/2510.20916)
  The aircraft collision avoidance system ACAS X takes "a variety of surveillance
  sources". Its tracking module must integrate them "both across time and across
  different surveillance sources", before the avoidance logic decides.
  **Why:** the oldest working last-minute system splits sensing (a tracker that fuses)
  from deciding.
- **Every drone chain in section 2 puts a tracker between detector and controller:**
  VOFOD (cluster tracker), Choi 2026 (EKF), Mandaokar 2026 (multiple models), Schilling
  2021 (multi-agent tracker), Pliska 2026 (tilt-aware Kalman filter).
- **Liu et al. 2017, *Learning End-to-end Multimodal Sensor Policies for Autonomous
  Navigation*** — [arXiv:1705.10422](https://arxiv.org/abs/1705.10422)
  Sensor Dropout: during RL training, switch off whole sensors at random, so the policy
  works with any subset. An extra loss stops jerks when a sensor drops. TORCS racing.
  **Why:** the direct recipe for "several sensors, any of them can fail".
- **Bohez et al. 2017, *Sensor Fusion for Robot Control through Deep RL*** —
  [arXiv:1703.04550](https://arxiv.org/abs/1703.04550)
  Networks that fuse several sensors and survive sensor failures at run time.
- **Neverova et al. 2015, *ModDrop*** — [arXiv:1501.00102](https://arxiv.org/abs/1501.00102)
  The original "drop a whole modality while training" trick.
- **Prakash, Chitta & Geiger 2021, *Multi-Modal Fusion Transformer for End-to-End
  Autonomous Driving*** — [arXiv:2104.09224](https://arxiv.org/abs/2104.09224)
  Camera and LiDAR fused with attention inside a driving policy. Support only.
- **Loquercio et al. 2021, *Learning High-Speed Flight in the Wild*** — Science Robotics,
  [arXiv:2110.05113](https://arxiv.org/abs/2110.05113)
  A teacher with full knowledge plans. A student sees only noisy depth, velocity and
  attitude and outputs 3 trajectories with their costs. 5–10 m/s in real forests, no
  real training data.
  **Why:** the standard way to train a sensor-only policy.
- **Song et al. 2023, *Learning Perception-Aware Agile Flight in Cluttered
  Environments*** — ICRA, [arXiv:2210.01841](https://arxiv.org/abs/2210.01841)
  RL teacher with full state, camera student by imitation. The policy learns to point
  its limited camera where it needs to see.
  **Why:** with a forward cone, where the drone looks is part of the task.

---

## 4. Learned dodging, one drone against a moving object

- **Kulkarni & Alexis 2024, *RL for Collision-free Flight Exploiting Deep Collision
  Encoding*** — [arXiv:2402.03947](https://arxiv.org/abs/2402.03947)
  Depth image → a small learned "collision code" that knows the robot's size → RL.
  **Why:** compress the sensor first, then train RL on the small code.
- **Vemprala, Mian & Kapoor 2021, *Representation Learning for Event-based Visuomotor
  Policies*** — [arXiv:2103.00806](https://arxiv.org/abs/2103.00806)
  The same split for event cameras, in simulation. Faster and more robust than feeding
  events as images.
- **Fan et al. 2025, *Flying in Highly Dynamic Environments with End-to-end Learning
  Approach*** — RA-L, [arXiv:2503.14352](https://arxiv.org/abs/2503.14352)
  Several LiDAR frames squeezed into a 2D obstacle map. One network outputs
  accelerations and dodges moving obstacles. Sim and real.
- **Xu et al. 2025, *Flow-Aided Flight Through Dynamic Clutters From Point To Motion*** —
  [arXiv:2511.16372](https://arxiv.org/abs/2511.16372)
  One LiDAR. A small distance map plus "point flow" over several frames go straight into
  RL. No detection, tracking or prediction step.
  **Why:** the other road from section 3: motion features with no tracker.
- **Kong et al. 2021, *Avoiding dynamic small obstacles with onboard sensing and
  computing on aerial robots*** — [arXiv:2103.00406](https://arxiv.org/abs/2103.00406)
  Classical: forward 3D LiDAR, map, A* search at 50 Hz onboard. Avoids bars down to
  20 mm at 2 m/s.
- **Xu et al. 2023, *A real-time dynamic obstacle tracking and mapping system ... with an
  RGB-D camera*** — ICRA, [arXiv:2209.08258](https://arxiv.org/abs/2209.08258)
  Classical: depth proposals, Kalman filter, motion prediction.

---

## 5. Avoiding each other with no radio

| paper | each agent senses | sensing limit | real robots |
|---|---|---|---|
| [CADRL](https://arxiv.org/abs/1609.07845), Chen et al. 2017 | exact position and velocity of others | none | — |
| [SA-CADRL](https://arxiv.org/abs/1703.08862), Chen et al. 2017 | same | none | ground robot among people |
| [GA3C-CADRL](https://arxiv.org/abs/1805.01956), Everett et al. 2018 | same, any number (LSTM) | none | ground vehicle |
| [Long et al. 2018](https://arxiv.org/abs/1709.10082) | raw 2D laser, last 3 frames | 180°, 4 m, 512 beams | — |
| [Batra et al. 2021](https://arxiv.org/abs/2109.07735) | K ≤ N−1 neighbours' relative state | K neighbours | small quadrotors |
| [Huang et al. 2024](https://arxiv.org/abs/2309.13285) | K nearest + obstacle distance field | K neighbours | quadrotors |
| [MACBF](https://arxiv.org/abs/2101.05436), Qin et al. 2021 | states of neighbours | local | — |
| [GCBF+](https://arxiv.org/abs/2401.14554), Zhang et al. 2024 | LiDAR rays | radius R = 0.5 (scene units) | drone swarm |
| [Huang et al. 2022](https://arxiv.org/abs/2203.02650) | depth image + IMU | camera FOV | — |
| [Zhang et al. 2025](https://arxiv.org/abs/2407.10648) | depth, pooled to 16 × 12 | camera FOV | yes |
| [Choi et al. 2026](https://arxiv.org/abs/2601.13657) | 360° LiDAR → up to 6 tracked neighbours + mask | 360°, small vertical FOV | 5 UAVs |
| [LEARN](https://arxiv.org/abs/2511.17765), Chiu et al. 2025 | low-res ToF | short range | 6 Crazyflies |
| [Schilling et al. 2021](https://arxiv.org/abs/2012.01245) | cameras + CNN + tracker, not RL | all around | 3 drones |
| [Petráček et al. 2021](https://doi.org/10.1088/1748-3190/abc6b3) | UVDAR, not RL | up to 15 m | outdoor swarm |

- **CADRL** learns a value network over the joint state of two agents. Its paths are
  over 26% better than ORCA's in time to goal.
- **SA-CADRL**: both the left-hand and the right-hand rule solve a head-on conflict;
  reciprocity does not need one unique rule. They teach one rule by penalising states
  where another agent comes from the unwanted side.
- **GA3C-CADRL** drops the assumption that others follow any rule, and reads any number
  of agents through an LSTM.
- **Long et al. 2018** train one shared policy on many robots across many scenarios at
  once, from raw laser scans.
- **Batra 2021 / Huang 2024** are the papers behind `quad-swarm-rl`. Batra samples each
  episode from one of six scenario types. Huang keeps a replay buffer of episodes that
  ended in collisions and restarts from them with a set probability.
- **MACBF** learns a safety certificate together with the policy. Trained on 8 agents,
  it runs with up to 1024.
- **GCBF+** does the same with a graph network on LiDAR rays inside a sensing radius, and
  flies a real drone swarm that swaps positions.
- **Zhang et al. 2025** train through a differentiable simulator: point-mass model,
  simple depth renderer. Swarm coordination with no radio and no central plan. Up to
  20 m/s in forests for one drone.
- **Huang et al. 2022**: each UAV runs the same policy, from depth and IMU, no radio.
  Simulation only. [Zhuang et al. 2024](https://arxiv.org/abs/2407.04064) improve how
  such a policy copes with scenes it has not seen.

---

## 6. Who dodges to which side

- **Fiorini & Shiller 1998, Velocity Obstacles** — IJRR,
  [10.1177/027836499801700706](https://doi.org/10.1177/027836499801700706).
  **van den Berg, Lin & Manocha 2008, RVO** — ICRA,
  [10.1109/ROBOT.2008.4543489](https://doi.org/10.1109/ROBOT.2008.4543489).
  **van den Berg et al. 2011, ORCA** —
  [10.1007/978-3-642-19457-3_1](https://doi.org/10.1007/978-3-642-19457-3_1).
  The geometric family. The HRVO paper explains the problem: if each robot assumes the
  other keeps its velocity, both avoid at once and oscillate.
- **Snape et al. 2011, HRVO** — T-RO, [10.1109/TRO.2011.2120810](https://doi.org/10.1109/TRO.2011.2120810)
  A robot that tries to pass on the wrong side must give the other full priority. On the
  right side it can count on the other's help. This removes the oscillation.
- **Zhou, Wang, Bandyopadhyay & Schwager 2017, Buffered Voronoi Cells** — RA-L,
  [10.1109/LRA.2017.2656241](https://doi.org/10.1109/LRA.2017.2656241)
  Deadlock is common in distributed avoidance, and they know of no method that provably
  avoids it without central computation. Their fix: a right-hand rule — every robot
  always detours to its right.
- **Wang, Ames & Egerstedt 2017, *Safety Barrier Certificates for Collisions-Free
  Multirobot Systems*** — T-RO, [10.1109/TRO.2017.2659727](https://doi.org/10.1109/TRO.2017.2659727);
  **Grover, Liu & Sycara 2019, *Deadlock Analysis and Resolution in Multi-Robot
  Systems*** — [arXiv:1911.09146](https://arxiv.org/abs/1911.09146)
  Control barrier functions keep robots apart but can deadlock. Grover shows a deadlock
  is a balance of forces on the edge of the safe set, and gives a decentralised way out.

**Our reading** for two drones that run one shared policy:

- In an exact head-on with equal speeds, both drones see the same thing in their own body
  frame — for any sideways offset. One policy then makes the same move for both. "My
  right" points opposite ways for the two drones, so a sideways dodge always passes. The
  right-hand rule of BVC and SA-CADRL appears without being asked for.
- That rule is the blind convention to watch. It breaks when the other drone does not run
  the same policy, or does not see the same thing.
- Up and down do not work that way: if both climb, they meet again.
- A policy that gives mirrored answers to mirrored inputs cannot pick a side in an exact
  head-on.
- Our eval uses mean actions, so the sampling noise that breaks ties in training is gone
  at eval.

---

## 7. Making a blind rule fail

- **Geirhos et al. 2020, *Shortcut Learning in Deep Neural Networks*** —
  [arXiv:2004.07780](https://arxiv.org/abs/2004.07780). The name of the failure:
  "always fly offset right" is a shortcut.
- **Tobin et al. 2017, *Domain Randomization*** — IROS,
  [10.1109/IROS.2017.8202133](https://doi.org/10.1109/IROS.2017.8202133). Train on many
  random variants so the real case is one more variant.
- **Kochenderfer, Kuchar et al. 2008, *Uncorrelated Encounter Model of the National
  Airspace System*** — MIT Lincoln Laboratory report ATC-345; journal version
  **Kochenderfer et al. 2010, *Airspace Encounter Models for Estimating Collision Risk*** —
  JGCD, [10.2514/1.44867](https://doi.org/10.2514/1.44867)
  To make one encounter: first sample the encounter geometry and start conditions from a
  joint distribution, then let a Markov model change turn rate, climb rate and speed
  during the encounter.
  **Why:** the template for our reset. Sample how the drones meet, not just where they
  start. Let the other drone change course during the encounter.
- **Jiang, Grefenstette & Rocktäschel 2020, *Prioritized Level Replay*** —
  [arXiv:2010.03934](https://arxiv.org/abs/2010.03934)
  Pick the next training scenario by how much there is still to learn from it (TD
  error). A curriculum appears by itself.
- **Dennis et al. 2020, *PAIRED*** — [arXiv:2012.02096](https://arxiv.org/abs/2012.02096)
  An adversary builds scenarios that the agent fails but that are solvable. They note
  that plain randomization cannot adjust difficulty, and pure worst-case search makes
  unsolvable scenarios.
- **Huang et al. 2024 collision replay** (section 5) — rare hard encounters come back
  more often.
- **Lee et al. 2020, *Adaptive Stress Testing*** — JAIR, [arXiv:1811.02188](https://arxiv.org/abs/1811.02188)
  RL searches for the most likely way a black-box system fails. Built for aircraft
  collision avoidance.
  **Why:** a test tool to find the encounters our policy fails.

**Our reading:**

- Only encounters with a miss distance under 0.80 m need a dodge (section 1). The share
  of such encounters in training sets how often the policy must dodge at all.
- With equal speeds and opposite headings, both drones always see the same thing
  (section 6). To check that the sensors are really used, break that symmetry: different
  speeds, approach angles away from 180°, different `d` or latency per drone, vertical
  offset, and another drone that does not dodge or turns during the encounter.

---

## 8. Follow a line and avoid

- **Meyer et al. 2020, *Taming an autonomous surface vehicle for path following and
  collision avoidance using deep RL*** — [arXiv:1912.08578](https://arxiv.org/abs/1912.08578)
  PPO steers a boat along a known path past fixed obstacles, sensed by several
  rangefinders. The agent sees its own reward weighting, from "stick to the path" to
  "avoid at any cost". Success 84–100% depending on that weighting.
  **Why:** our two goals, with range-limited sensors, and the trade-off as an input.
- **Falanga 2019** (section 1) assumes the robot keeps its forward speed and only moves
  sideways — the same rule as "stay on your line, keep your speed".

---

## 9. Simulating sensors cheaply

- **Kulkarni, Rehberg & Alexis 2025, *Aerial Gym Simulator*** —
  [arXiv:2503.01471](https://arxiv.org/abs/2503.01471)
  GPU ray casting for depth and segmentation, with depth-based RL examples that transfer
  to real drones.
- **Schuck et al. 2026, *Crazyflow*** — [arXiv:2606.01478](https://arxiv.org/abs/2606.01478)
  JAX drone simulator. Sampling-based obstacle avoidance at over half a billion steps per
  second.
- **Zhang et al. 2025** (section 5): a point-mass model and depth pooled to 16 × 12 were
  enough for zero-shot transfer.
- **GCBF+** models LiDAR as evenly spaced rays inside a radius; **Long 2018** uses 512
  beams × 3 frames; **Schilling 2021** gets distance from apparent size.

**Our reading:** our drones are spheres of radius 0.40 m. A ray–sphere hit is a closed
formula, so a LiDAR or ToF ray costs a few operations per drone. A camera can be modelled
without rendering as bearing plus apparent size inside a FOV cone. Noise, misses, false
alarms and delay can be drawn per step. What this loses: background clutter and light,
which Det-Fly and Schilling 2021 show is what makes real camera detection hard.

---

## 10. Baselines

| baseline | needs | note |
|---|---|---|
| artificial potential field, [Khatib 1986](https://doi.org/10.1177/027836498600500106) | position of the other drone | already a seat in `seats.py`. Reactive methods are prone to deadlock (section 6) |
| VO / RVO / ORCA / HRVO | position and velocity of the other drone | RVO-type methods assume the other drone also avoids |
| control barrier function filter, [Ames et al. 2019](https://doi.org/10.23919/ECC.2019.8796030) | position and velocity of the other drone | on top of the line-following cascade. Mandaokar 2026 does this with radar |
| sidestep rule | bearing only | dodge away from the bearing, right when dead ahead: the LGMD rule (Zhao 2019) plus the right-hand rule (BVC) |

---

## 11. What the field agrees on

1. Sensing range, latency and sideways acceleration together set the fastest safe speed
   (Falanga 2019, Mandaokar 2026).
2. Every real drone-detection chain has a tracker between the detector and the
   controller (section 3).
3. Learned avoidance with no radio works when all agents run the same policy (Long 2018,
   Batra 2021, Huang 2022, Zhang 2025).
4. Either side works. Both agents must pick compatible sides. Papers fix it with a rule
   (SA-CADRL, BVC) or by giving way on the wrong side (HRVO).
5. With partial sensing, papers add memory — frame stacks (Long 2018, Fan 2025, Xu 2025)
   or an LSTM (GA3C-CADRL) — and masks for missing neighbours (Choi 2026).
6. To train on poor sensors, a teacher sees the truth and a student sees the sensors
   (Loquercio 2021, Song 2023).
7. To survive a lost sensor, switch sensors off at random while training (Sensor Dropout,
   ModDrop, Bohez 2017).

## 12. Gaps

- No paper found trains two quadrotors to dodge each other head-on, with range-limited
  input from several sensor types and no radio. Closest: Choi 2026 (one LiDAR,
  flocking), Zhang 2025 (one depth camera, swarm), LEARN (ToF, nano drones).
- Air-to-air detection data is thin at short range. Det-Fly starts at 10 m. The
  multi-sensor sets (MMAUD, Svanström) look up from the ground.
- Most multi-agent RL papers give each agent the exact state of its neighbours (CADRL,
  Batra 2021, MACBF).
- Onboard radar on small drones is new. Mandaokar 2026 call theirs the first such system
  for fast dodging.

## 13. Open questions for the grilling

1. Sensor shape: a forward cone, all around, or one of each?
2. Which sensors, and what does each give: bearing, range, speed, full 3D?
3. What `d` per sensor: near the section-1 numbers, or well above them?
4. Who is the other drone: the same policy, a scripted drone that never dodges, or a mix?
5. Are dodges up and down allowed?
6. What is random, and over what range: sideways and vertical offset, approach angle,
   speeds, meeting time, sensor noise, misses, false alarms, sensor dropout, course
   changes of the other drone?
7. Memory: a frame stack, a recurrent policy, or a tracker in front of the policy?
8. A teacher-student setup, or RL directly on the sensor input?
9. Which metrics: collisions, smallest gap, distance from the line, time lost, success?
10. Which baseline sits in the other seat?
