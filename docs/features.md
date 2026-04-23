# Statcast Feature Glossary

Full reference for all columns in the raw Statcast dataset. Features marked with ⭐ are used in the classifier model.

---

## Pitch Identification

| Feature | Description |
|---|---|
| `pitch_type` | Pitch type code (FF, SL, CU, etc.) — the classification target |
| `pitch_name` | Full pitch name (e.g. "4-Seam Fastball") |
| `game_date` | Date the game was played |
| `game_pk` | Unique game identifier |
| `game_type` | Game type (R=Regular Season, P=Postseason, etc.) |
| `game_year` | Season year |
| `at_bat_number` | At-bat number within the game |
| `pitch_number` | Pitch number within the at-bat |
| `sv_id` | Unique pitch identifier |

---

## Players & Teams

| Feature | Description |
|---|---|
| `player_name` | Pitcher's full name (Last, First) |
| `pitcher` | Pitcher's MLBAM ID |
| `batter` | Batter's MLBAM ID |
| `p_throws` ⭐ | Pitcher handedness (R/L) |
| `stand` | Batter handedness (R/L) |
| `home_team` | Home team abbreviation |
| `away_team` | Away team abbreviation |
| `fielder_2` through `fielder_9` | MLBAM IDs of fielders on the play |

---

## Velocity & Spin

| Feature | Units | Description |
|---|---|---|
| `release_speed` ⭐ | mph | Pitch speed at release |
| `effective_speed` | mph | Perceived speed adjusted for release extension |
| `release_spin_rate` ⭐ | rpm | Spin rate at release |
| `spin_axis` | degrees | Axis around which the ball spins (0–360°) |
| `spin_dir` | — | Deprecated spin direction field |
| `spin_rate_deprecated` | — | Deprecated spin rate field |

---

## Movement (Pitch Break)

| Feature | Units | Description |
|---|---|---|
| `pfx_x` ⭐ | ft | Horizontal movement vs. no-spin trajectory. Negative = left, Positive = right (catcher's view) |
| `pfx_z` ⭐ | ft | Vertical movement vs. no-spin trajectory. Positive = rise (backspin), Negative = drop (topspin) |
| `api_break_z_with_gravity` | inches | Total vertical drop including gravity |
| `api_break_x_arm` | inches | Horizontal break from pitcher's arm-side perspective |
| `api_break_x_batter_in` | inches | Horizontal break toward the batter |
| `break_angle_deprecated` | — | Deprecated break angle field |
| `break_length_deprecated` | — | Deprecated break length field |

---

## Release Point

| Feature | Units | Description |
|---|---|---|
| `release_pos_x` ⭐ | ft | Horizontal hand position at release. Negative = right side, Positive = left side |
| `release_pos_z` ⭐ | ft | Vertical hand height at release (~5–6.5 ft) |
| `release_pos_y` | ft | Distance from home plate at release (~55 ft) |
| `release_extension` ⭐ | ft | How far toward the plate the pitcher extends before releasing |
| `arm_angle` | degrees | Pitcher's arm slot angle at release |

---

## Plate Location

| Feature | Units | Description |
|---|---|---|
| `plate_x` ⭐ | ft | Horizontal location as pitch crosses plate. 0 = center |
| `plate_z` ⭐ | ft | Vertical height as pitch crosses plate |
| `sz_top` | ft | Top of batter's strike zone |
| `sz_bot` | ft | Bottom of batter's strike zone |
| `zone` | 1–14 | Strike zone region (1–9 = in zone, 11–14 = out of zone) |

---

## Ball Physics (At Release)

| Feature | Units | Description |
|---|---|---|
| `vx0` | ft/s | Horizontal velocity at release |
| `vy0` | ft/s | Velocity toward plate at release |
| `vz0` | ft/s | Vertical velocity at release |
| `ax` | ft/s² | Horizontal acceleration |
| `ay` | ft/s² | Acceleration toward plate |
| `az` | ft/s² | Vertical acceleration |

---

## Count & Game Situation

| Feature | Description |
|---|---|
| `balls` | Ball count before the pitch |
| `strikes` | Strike count before the pitch |
| `outs_when_up` | Number of outs when the pitch was thrown |
| `inning` | Inning number |
| `inning_topbot` | Top or bottom of the inning |
| `on_1b` / `on_2b` / `on_3b` | MLBAM ID of runner on base (null if empty) |

---

## Pitch Outcome

| Feature | Description |
|---|---|
| `type` | Simplified outcome: S=Strike, B=Ball, X=In play |
| `description` | Detailed outcome (e.g. "swinging_strike", "called_strike", "ball") |
| `events` | At-bat ending event (e.g. "strikeout", "home_run") — null if at-bat continues |
| `des` | Text description of the play |
| `zone` | Strike zone location (1–14) |

---

## Batted Ball

| Feature | Units | Description |
|---|---|---|
| `launch_speed` | mph | Exit velocity off the bat |
| `launch_angle` | degrees | Vertical angle of the ball off the bat |
| `launch_speed_angle` | — | Combined speed/angle category (1–6) |
| `hit_distance_sc` | ft | Projected distance of the batted ball |
| `hc_x` / `hc_y` | pixels | Hit coordinates on the field diagram |
| `hit_location` | 1–9 | Fielder position where ball was hit |
| `bb_type` | — | Batted ball type (ground_ball, fly_ball, line_drive, popup) |

---

## Batter Swing Metrics

| Feature | Units | Description |
|---|---|---|
| `bat_speed` | mph | Speed of the bat at contact |
| `swing_length` | ft | Length of the swing path |
| `attack_angle` | degrees | Bat angle at contact |
| `attack_direction` | degrees | Horizontal direction of the swing |
| `swing_path_tilt` | degrees | Tilt of the swing plane |
| `intercept_ball_minus_batter_pos_x_inches` | inches | Horizontal bat-ball contact offset |
| `intercept_ball_minus_batter_pos_y_inches` | inches | Depth bat-ball contact offset |
| `hyper_speed` | — | Composite bat speed metric |

---

## Expected Statistics

| Feature | Description |
|---|---|
| `estimated_ba_using_speedangle` | Expected batting average based on exit velo + launch angle |
| `estimated_woba_using_speedangle` | Expected wOBA based on exit velo + launch angle |
| `estimated_slg_using_speedangle` | Expected slugging based on exit velo + launch angle |
| `woba_value` | Actual wOBA value of the play |
| `woba_denom` | wOBA denominator (plate appearance weight) |
| `babip_value` | BABIP value of the play |
| `iso_value` | Isolated power value of the play |

---

## Win Probability & Run Expectancy

| Feature | Description |
|---|---|
| `delta_home_win_exp` | Change in home team win probability from the pitch |
| `delta_run_exp` | Change in run expectancy from the pitch |
| `delta_pitcher_run_exp` | Change in run expectancy attributed to the pitcher |
| `home_win_exp` | Home team win probability before the pitch |
| `bat_win_exp` | Batting team win probability before the pitch |
| `home_score_diff` | Score difference (home minus away) |
| `bat_score_diff` | Score difference from batting team's perspective |

---

## Score

| Feature | Description |
|---|---|
| `home_score` / `away_score` | Score before the pitch |
| `bat_score` / `fld_score` | Batting/fielding team score before the pitch |
| `post_home_score` / `post_away_score` | Score after the pitch |
| `post_bat_score` / `post_fld_score` | Batting/fielding team score after the pitch |

---

## Fielding Alignment

| Feature | Description |
|---|---|
| `if_fielding_alignment` | Infield alignment (Standard, Shift, etc.) |
| `of_fielding_alignment` | Outfield alignment (Standard, Strategic, etc.) |

---

## Player Age & Usage

| Feature | Description |
|---|---|
| `age_pit` / `age_bat` | Pitcher/batter age during the season |
| `age_pit_legacy` / `age_bat_legacy` | Legacy age calculation |
| `n_thruorder_pitcher` | How many times through the batting order the pitcher has gone |
| `n_priorpa_thisgame_player_at_bat` | Number of prior plate appearances by the batter in the game |
| `pitcher_days_since_prev_game` | Days of rest for the pitcher |
| `batter_days_since_prev_game` | Days since the batter last played |
| `pitcher_days_until_next_game` | Days until the pitcher's next appearance |
| `batter_days_until_next_game` | Days until the batter's next game |

---

## Deprecated Fields

These fields are kept for backwards compatibility but should not be used.

| Feature | Description |
|---|---|
| `spin_dir` | Replaced by `spin_axis` |
| `spin_rate_deprecated` | Replaced by `release_spin_rate` |
| `break_angle_deprecated` | Replaced by `api_break_x_arm` |
| `break_length_deprecated` | Replaced by `api_break_z_with_gravity` |
| `tfs_deprecated` | Deprecated timestamp field |
| `tfs_zulu_deprecated` | Deprecated UTC timestamp field |
| `umpire` | No longer populated |
