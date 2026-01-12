# Predicting Quarterback Ball Placement Pre-Throw: A Spatio-Temporal Modeling Approach
Sahil Shah, _Data Scientist_

### Introduction
The modern NFL is an environment of extreme information asymmetry. For a defense, preparation is an exhaustive cycle of film study, tendency analysis, and tactical installation designed to minimize the offensive "edge." However, as offensive strategies evolve to become more deceptive,
the margin for defensive error has narrowed. Traditionally, quarterback play was predicated on coverage identification—reading single-high versus split-safety shells to determine a "side" of the field. 
As noted by veteran quarterback Kirk Cousins, a fundamental shift occurred as defenses became proficient at disguising these shells. This forced an evolution in offensive coaching toward "pure progressions"—a sequential, timing-based read ($1 \to 2 \to 3 \to 4 \to 5$) that prioritizes internal rhythm
over defensive reaction. While elite quarterbacks, notably Aaron Rodgers, Matthew Stafford, and Joe Burrow, still utilize hybrid methods—blending pre-snap recognition with rapid post-snap processing—this shift toward standardized movement patterns provides a unique opportunity for machine learning. By analyzing the "battlefield" as a dynamic system of 22 interacting agents,
we can identify latent patterns in player kinematics that signal a quarterback’s intent before the ball is released.

<p align="center">
  <a href=https://www.youtube.com/shorts/wtut3dFo84k>
    <img src="https://github.com/user-attachments/assets/ab419a6b-0dbb-4d5d-9f22-4f997a71be1a" width="394" height="400" alt="Watch the video">
  </a>
</p>

### Abstract
This research proposes a predictive framework designed from a defensive perspective to forecast quarterback ball placement during the pre-throw phase of an NFL play. By leveraging high-resolution tracking data, this model aims to assist defensive coaching staffs in quantifying offensive tendencies,
identifying positional vulnerabilities, and optimizing situational defensive strategies. Using a multi-layer Transformer architecture, the model maps the spatial relationships of all 22 players to output predicted target coordinates ($X, Y$).
The results demonstrate that, despite the modern shift toward pure progression-based passing, pre-snap alignment and early post-snap kinematics remain highly predictive indicators of ball placement.

### Data 
The dataset utilized in this study was sourced from the 2026 NFL Big Data Bowl, comprising tracking data from Weeks 1–18 of the 2023–2024 season. The explanation of the features is provided here: https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/data. The raw data consists of approximately 5 million frame-level samples (recorded at 10 Hz), which were consolidated into a play-centric format. 
To capture the full spatial context of each play, data for all 22 players were synchronized, resulting in a processed dataset of 350,000 samples with approximately 200 feature dimensions per frame.

Data Filtering and Preprocessing:
  - Scope: The study focuses on pass attempts(excluding incompletions for Go, Corner, and Post routes due to large variance in ball placement compared to shorter and intermediate routes), excluding scrambles, throwaways, and batted balls.
  - Leakage Prevention: A 80/20 train-test split was implemented based on Game ID to ensure that no specific game-day conditions or repeated matchups influenced the test results.
  - Oversampling: To address the inherent volatility in deep-field targets, plays involving vertical routes (Go, Post, and Corner) were oversampled (adding ~40,000 samples) on the y axis to ensure the model captured the high-variance kinematics of long-distance throws.
  - Feature Engineering: Inputs include velocity vectors, instantaneous acceleration, and "leverage angles"—the relative positioning between a defender and a receiver.
    
### Model Architecture
To simulate the spatial reasoning and decision-making processes inherent in professional football, a Transformer-based architecture was employed. The Transformer’s self-attention mechanism is uniquely suited for this task, as it can weigh the relative importance of all 22 players simultaneously, preserving the spatial relationships between offensive threats and defensive counters.

Key Architectural Features:
  - High-Dimensional Embeddings: Each player’s state is projected into a higher-dimensional vector space. I implemented distinct positional embeddings for offensive and defensive units to learn the unique movement signatures of different roles (e.g., the burst of a wide receiver versus the backpedal of a cornerback).
  - Temporal History: The model utilizes a "look-back" window of previous frames, allowing the Transformer to interpret changes in velocity and direction as a sequence rather than isolated snapshots.
  - Loss Function Optimization: I utilized a combination of Huber Loss and Root Mean Square Error (RMSE). Huber Loss was selected for its robustness against outliers (e.g., broken plays), while an exponential ramp penalty was applied to sideline and endzone boundaries to enforce "stay-in-bounds" logic.
  - Weighting Schema: The training objective was weighted (77% standard / 23% long-ball) and 1.2x on completions, 0.8 on incompletions, and 1.1 on interceptions to prioritize accuracy on high-leverage, deep-field attempts and more accurate throws.

The final output is a 6-layer Transformer that generates a continuous coordinate prediction $[X, Y]$ for each frame, providing a real-time "heat map" of the quarterback's most likely target.

### Model Performance

<p align="center"><img width="645" height="510" alt="image" src="https://github.com/user-attachments/assets/786ddb24-5fc4-4011-9d78-e5b3e60b7950" /> </p>

This model, yielding an average RMSE of 9.05 yards, a Huber Loss of 8.61 yards for all frames and a median RMSE of 6.2 yards for peak-accuracy frames, performs within a competitive margin of these industry leaders. Notably, SOTA models are typically trained on the full proprietary Next Gen Stats (NGS)
corpus—approximately 8x more data than was available for this study—and incorporate high-resolution features such as throwing direction and player-specific 'speed signatures' + player context. While I am satisfied that the model captures the primary spatial intent of the play within a 6-yard radius
during the high-entropy window of the pocket collapse, I believe that closing the remaining gap would require the integration of the broader contextual datasets used by the NFL’s primary analytics partners. In order to further break down model performance, I investigated in which scenarios the model could predict ball placement way before the quarterback gets in his throwing motion. This is crucial as the defense can learn and game plan based off early offensive tendencies.
The next sections discuss in which situations the model does well, spotlight team + route based tendencies, and spotlight player + position + route based tendencies. 

### Team Based Tendencies based on Route
I analyzed plays/teams where my prediction model achieved confidence(solved the play/ball location, < 8 yards prediction error, in under 60% of the dropback) versus plays where it remained "Reactive" until close to the throw (avg. 3.00s solution time). Here are the top 3 team + route combinations:

Kansas City GO Route: On average, the model was able to predict ball placement within 8 yards when the targeted receiver was running a Go route in 1.63 seconds after the snap. The model knew within 0.33 seconds after the snap where the ball was going within 4 yards on 50% of those GO routes. This highlights the importance of pre-snap and early post-snap reads, as less emphasis is being placed on them as explain by Kirk Cousins and many others.

Core identity: Kansas City’s confident targeted GO routes show up when:
- The opposing secondary over-rotates and declares leverage early, especially on the targeted outside receiver
- Opposing Corner & Linebacker Orientation Is a Massive Tell
- 73% GO balls on first down
- Second and third-level defenders are compressed laterally and closer to the LOS.
- Average Point Differential +6.91

For example, as seen in Week 2 of Chiefs vs. Jaguars, Mahomes sees the 1-on-1 matchup between Justin Watson and Tyson Campbell, with a single high safety in between the numbers, he knows he can take a shot. Its 1st down, KC are up 5, and majority of second level defenders are close to the box. The model knows exactly where Mahomes is going with the football, way before he even catches the ball from the snap.

<p align="center">
  <img src="https://github.com/user-attachments/assets/bdf49496-3fd0-4af8-bd91-dffeead8689c" alt="KC_GO_1" width="600">
</p>


<p align="center"><img width="374" height="315" alt="image" src="https://github.com/user-attachments/assets/5f842bcd-ae6c-4aa4-a393-513b9ed87e54" /> <img width="421" height="370" alt="image" src="https://github.com/user-attachments/assets/cabc4ac1-de54-4780-b0b2-697a5bde402e" /></p>


Tampa Bay CORNER Route: On average, the model identified the target within 8 yards on 100% of Tampa Bay’s targeted CORNER routes in 1.68 seconds after the snap. The model knew within 0.23 seconds after the snap where the ball was going within 4 yards on 47% of those CORNER routes. This continues to highlight the importance of pre-snap and early post-snap reads.

Core identity: TB’s confident corner routes come when:
- The opposing flat defender is physically misaligned
- The defense is over-committed to the run fit and high-low stress concepts.
- Safeties are stepping into run support or inside zones
- Targeted Receiver is typically in a tight split and part of either a bunch/stack.

For example, as seen in Week 3 of Eagles vs. Bucs, all parts of TB's Corner route criteria are fulfilled, and the model knows exactly where the ball is going even before Mayfield catches the snap.

<p align="center">
  <img src="https://github.com/user-attachments/assets/d325da62-73ea-44c9-8f8d-b05193a7150a" alt="KC_GO_1" width="600">
</p>

<p align="center"><img width="374" height="315" alt="image" src="https://github.com/user-attachments/assets/06831b25-28c4-4548-a256-6a6f96576efd"/> <img width="421" height="370" alt="image" src="https://github.com/user-attachments/assets/04722334-3d74-41c2-8bda-8d27138401f6" /></p>

Arizona ANGLE Route: On average, the model identified ball placement on Arizona’s targeted ANGLE routes within 7.5 yards on 100% of plays in 1.4 seconds. The model knew within 0.39 seconds after the snap where the ball was going within 4.2 yards on 50% of those ANGLE route plays. Once again, the model results show that pre snap/early post snap reads should be given more emphasis as they can predict ball placement in such a manner.

Core identity: ARI’s confident ANGLE routes come when:
- QB does not scan, he almost immediately commits: The change in absolute orientation from the 80% of the dropback to first frame post-snap, spikes (206.77°), showing a full torso turn.
- Distance between QB and all eligible receivers jumps further in the 0-80% window of the dropback than in non-confident predictions.
- Eye Contact as an Early Contract: Visual alignment stabilizes almost immediately. WR eye-contact angle jumps to 0.38 and stays stable.
  
<p align="center"><img width="535" height="462" alt="image" src="https://github.com/user-attachments/assets/8e2fc916-24c0-4697-aa4e-b9f496338306" /></p>

### Player + Route Based Tendencies

<p align="center"><img width="1542" height="814" alt="image" src="https://github.com/user-attachments/assets/59f6588f-9090-4d9e-bcef-4f25e51090d9" /> </p>

Players with the Highest Pre-Throw Model Confidence rate of Whether they are Targeted

<a href="https://github.com/sashah12/NFL-Big-Data-Bowl-2026/blob/main/Model%20Confident%20Player%20Targets%20vs.%20Player%20Non-Targets%20Baseline%20Tendencies.PNG">Model Confident Player Targets vs Player Non Targets Baseline</a>

1. Dallas Goedert (Eagles TE - Position 2, FLAT Route)
Goedert's primary tell is an immediate change in spatial relationships and directional commitment in the 0-40% window.
Insight 1: The "Burst" Signal (0-40%): The rate at which Goedert increases his separation (off2_def_proximity) from the nearest defender in the first 0.5s to 2.0s is significantly faster (+3.5 yards/sec faster rate of separation gain) on targeted plays than when he runs a decoy route.
Insight 2: Immediate Directional Commitment (0-40%): His change in direction (off2_dir) in the early window is 16 degrees larger when targeted. He commits his path instantly, rather than stuttering or playing neutral.
Defensive Gameplan: Do not rely on pre-snap alignment. Key the H-back's immediate release in the first second of the play. If he bursts away from the nearest defender at full speed and commits to an outside angle instantly, jump the flat route.
2. Tyler Higbee (Rams TE - Position 2, FLAT/SLANT)
Higbee is predictable because he rapidly achieves an optimal spatial alignment relative to the target line early in the down.
Insight 1: Rapid Angle Alignment (0-40%): The rate at which his angle aligns with the intended target line (off2_target_angle_offset_sin) is much faster on targeted plays (large negative bar). He quickly squares himself up to the line of scrimmage.
Insight 2: Rapid Leverage Gain: The rate at which he gains outside leverage (off2_leverage_angle) is faster early in the play. He works harder in the first 40% window to get outside the defender.
Defensive Gameplan: The window for deception closes very quickly (before the 1-second mark). Play aggressive trail coverage and force him to maintain vertical depth longer, disrupting his rapid angle alignment.
3. Adam Thielen (Panthers WR - Position 4, HITCH/OUT/SLANT)
Thielen's tell is a change in how much space he demands from the defense early in the down.
Insight 1: Demanding Space Early: The rate at which he separates from the defense is much faster in the first two seconds (off4_def_proximity, nearest_def_dist). When targeted, he works harder to get clean releases instantly.
Defensive Gameplan: Eliminate his "space gain" tell by consistently playing jam/press coverage at the line. Do not allow him a clean release into the 0.5-second window, which disrupts his ability to execute the high-rate separation that signals intent.
4. Derrick Henry (Titans RB - Position 3, ANGLE/FLAT)
Henry's predictability stems from visual coordination and efficient pathing that develops quickly post-snap.
Insight 1: Instant QB Lock-On: The rate at which the QB-WR visual contact angle stabilizes (off3_wr_eye_contact_angle) is faster on targeted plays. This visual alignment happens instantly.
Insight 2: Efficient Pathing: The change in his path angle offset is heavily negative (off3_target_angle_offset), meaning he aggressively aligns his movement vector with the ball's likely landing spot faster than when he's a decoy.
Defensive Gameplan: LBs must ignore the backfield entirely and key the QB's eyes immediately post-snap. If the QB’s visual angle stabilizes toward Henry in the first second, jump the route.
5. Puka Nacua (Rams WR - Position 5, SLANT/FLAT/HITCH)
Nacua's tells are driven by changes in defensive depth manipulation.
Insight 1: Depth Manipulation (0-40%): The rate at which he manipulates defender depth (off5_cb_depth, s_depth) is rapid. When targeted, the distance metrics quickly shift to indicate shallow coverage, a strong tell that the defense has committed.
Insight 2: Boundary Pressure Rate: The change in his boundary proximity is faster when targeted, indicating he instantly bursts toward the sideline boundary constraint.
Defensive Gameplan: Disguise defensive depth. Do not give a consistent shallow-coverage look post-snap, which forces the QB to verify Nacua's intent and slows down the play development.


<a href="https://github.com/sashah12/NFL-Big-Data-Bowl-2026/blob/main/Model%20Confident%20Player%20Targets%20vs.%20League%20Targets%20Baseline%20Tendencies.PNG">Model Confident Player Targets vs League Targets Baseline</a>

1. Derrick Henry (RB - Position 3, FLAT/SLANT)
Henry's predictability comes from immediate movement efficiency and visual alignment signals.
Flat Routes: Henry shows a major deviation in off3_dir (direction) and off3_path_angle_cos. This indicates he runs a significantly more direct, efficient angle immediately upon release in the flat than the average NFL running back.
Slant Routes: His off3_wr_eye_contact_angle is a key positive deviation. The QB achieves a quicker, more stable visual lock on Henry when the slant is coming, a strong pre-throw tell.
2. Tyler Higbee (TE - Position 2, FLAT/OUT)
Higbee is a confident target when his angle is acutely defined and he demands leverage early.
Flat/Out Routes: Higbee has large negative deviations in off2_target_angle_offset_sin. He instantly commits to a very horizontal, sideline-bound path, removing all vertical ambiguity early in the route compared to the league average TE.
Overall: He consistently creates more def_proximity and off2_y separation, showing he aggressively works his initial release to secure space instantly.
3. Josh Jacobs (RB - Position 3, ANGLE/FLAT)
Jacobs is a confident target when he instantly establishes clear route geometry.
Angle/Flat Routes: He exhibits significant positive deviations in off3_path_angle_cos. Jacobs aligns his movement vector with the play direction faster and more decisively than the average running back.
Overall: His off3_speed_change is less volatile (small deviation), suggesting he maintains a stable speed profile on confident targets, providing fewer "speed-change" tells.
4. Dallas Goedert (TE - Position 2, FLAT/SLANT/OUT)
Goedert relies heavily on immediate spatial manipulation and defined leverage points.
Flat/Slant/Out Routes: Goedert's positive deviation in off2_def_proximity metrics means he consistently works to get more separation earlier in the down than the average tight end running these routes.
Overall: His off2_target_angle_offset deviation is negative, indicating he runs highly acute, "cornered" routes that eliminate ambiguity for the QB and the model.
5. CeeDee Lamb (WR - Position 2, SLANT/HITCH/OUT)
Lamb's tells are driven by space creation and defensive reaction time.
Slant/Hitch/Out Routes: He shows large positive deviations in off2_def_proximity and nearest_def metrics. Lamb demands and achieves more open space from defenders faster than the league average, a strong signal he's the primary read.
Overall: His vertical velocity (off2_v_y) deviation is lower, suggesting less fluctuation in vertical depth early in the down compared to reactive targets.
6. Puka Nacua (WR - Position 1/5, SLANT/FLAT/ANGLE)
Nacua's predictability stems from defined defensive depth and predictable pressure.
Slant/Flat/Angle Routes: He consistently achieves shallower defensive depth (cb_depth, s_depth are negative deviations). The defense is sitting on his routes when he is targeted confidently.
Overall: His boundary_pressure metrics show more predictable pressure relative to the sideline on these routes compared to the average WR target.
7. DeVonta Smith (WR - Position 5, FLAT/SLANT/HITCH)
Smith’s tells involve movement efficiency and reduced defensive reaction time.
Flat/Slant/Hitch Routes: His off5_dir deviation is low, suggesting a stable movement direction that doesn't change much as the play develops.
Overall: He demonstrates less early route variation (off5_triangle_depth_variance), pointing to efficient, non-deceptive route running on confident targets.
8. Adam Thielen (WR - Position 4, HITCH/OUT/SLANT)
Thielen uses immediate space to signal intent.
Hitch/Out/Slant Routes: He consistently has a larger off4_wr_dist_to_sideline when targeted confidently, indicating he exploits maximum width in the formation immediately post-snap.
Overall: His def_proximity_at_0.5s is positive, highlighting that early space is the primary tell for his confident targets.

### Improvements
There is still a large opportunity for improvement. The SOTA models utilize years worth of data, player context, and other high resolution data. But with the current data I have, I can still improve the model more, focusing more on reducing outliers on long balls as well as implementing an intended target prediction-throw placement can be affected by incompletions in the training data. Additionally, other model architectures, such as Graph Neural Networks, have shown to be strong and have an opportunity to be explored.
