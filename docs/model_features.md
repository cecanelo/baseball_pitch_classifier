# Model Features — Detailed Reference

This document explains the 10 features used to train the pitch classifier. Each feature was selected because it captures a physically meaningful aspect of how a pitch is thrown and travels to the plate.

---

## Why These Features?

A pitch can be characterized by four questions:
1. **How fast is it?** → `release_speed`, `effective_speed`
2. **How does it move?** → `pfx_x`, `pfx_z`, `release_spin_rate`
3. **Where does it come from?** → `release_pos_x`, `release_pos_z`, `release_extension`
4. **Where does it end up?** → `plate_x`, `plate_z`

Together these features give the model a complete picture of the pitch from the moment it leaves the hand to the moment it crosses the plate.

---

## Velocity

### `release_speed`
- **Units:** mph
- **Range:** ~65–105 mph
- **What it captures:** Raw speed at the moment of release — the most immediately recognizable characteristic of a pitch.
- **Why it matters for classification:** Fastballs (~90–100 mph) and breaking balls (~70–85 mph) occupy different speed ranges. However, speed alone is insufficient — a Kershaw fastball (~91 mph) overlaps with a Wheeler changeup (~88 mph), so the model needs additional features to disambiguate.

---

## Spin

### `release_spin_rate`
- **Units:** rpm (revolutions per minute)
- **Range:** ~1,000–3,000 rpm
- **What it captures:** How fast the ball spins as it leaves the hand. Spin interacts with air resistance to create movement — more spin generally means more movement.
- **Why it matters for classification:** The splitter (FS) is thrown with intentionally low spin (~1,200 rpm) causing it to drop sharply. Fastballs and curveballs share similar spin rates but in opposite directions, so spin rate alone doesn't separate them — spin axis (not included) would be needed for that.

---

## Movement

These two features are the most powerful separators in the dataset. They measure how much the pitch deviated from a theoretical straight path thrown with zero spin, with only gravity acting on it.

### `pfx_x`
- **Units:** feet
- **Range:** ~-2.0 to +2.0 ft
- **What it captures:** Horizontal deflection — how far left or right the pitch moved beyond what gravity alone would produce.
- **Sign convention (catcher's perspective):**
  - Negative → movement toward the left
  - Positive → movement toward the right
- **Why it matters for classification:** A right-handed pitcher's sweeper (ST) breaks sharply to the right (+1.5 ft), while his sinker (SI) runs back to the left (-1.0 ft). However, handedness flips the sign — a left-handed pitcher's sweeper breaks left. This is why `p_throws` is included alongside this feature.

### `pfx_z`
- **Units:** feet
- **Range:** ~-2.0 to +2.0 ft
- **What it captures:** Vertical deflection — how much the pitch "rose" or dropped beyond what gravity alone would produce.
- **Sign convention:**
  - Positive → backspin fighting gravity ("rise") — fastballs
  - Negative → topspin accelerating the drop — curveballs
- **Why it matters for classification:** This is the single strongest separator in the dataset. Fastballs (+1.3 to +1.8 ft) and curveballs (-0.8 to -1.4 ft) occupy opposite ends of the range with minimal overlap. Combined with `pfx_x`, these two features create a 2D "movement fingerprint" unique to each pitch type.

---

## Release Point

Where the pitcher's hand is at the moment of release. Consistent release points across pitch types are a key deception tool — batters cannot identify the pitch early from the arm action alone.

### `release_pos_x`
- **Units:** feet
- **Range:** ~-3.0 to +3.0 ft
- **What it captures:** Horizontal position of the hand at release, measured from the center of the pitcher's rubber.
- **Why it matters for classification:** Release point varies slightly between pitch types for some pitchers — a pitcher may drop their arm slot for a curveball. However, the dominant signal in this feature is pitcher handedness (right-handers release from the negative side, left-handers from the positive side), which is why `p_throws` is essential context for this feature.

### `release_pos_z`
- **Units:** feet
- **Range:** ~5.0 to 6.5 ft
- **What it captures:** Height of the hand at release. Reflects the pitcher's arm slot — over-the-top vs. sidearm.
- **Why it matters for classification:** Some pitchers drop their arm slot for specific pitch types, creating a slightly different release height. Also affects the downward angle of the pitch into the strike zone.

### `release_extension`
- **Units:** feet
- **Range:** ~5.5 to 7.5 ft
- **What it captures:** How far toward home plate the pitcher strides and extends before releasing the ball.
- **Why it matters for classification:** Extension reduces the effective distance to the plate, giving the batter less reaction time. Pitchers often extend differently for different pitch types — a fastball may come with maximum extension while a changeup has slightly less, contributing to the deception.

---

## Plate Location

Where the pitch crosses home plate, from the catcher's perspective.

### `plate_x`
- **Units:** feet
- **Range:** ~-3.0 to +3.0 ft (0 = center of plate)
- **What it captures:** Horizontal location as the pitch crosses the plate.
- **Why it matters for classification:** Pitch location is partly a consequence of pitch type — cutters naturally finish glove-side, sinkers finish arm-side. However, location is also a strategic choice by the pitcher, making it a weaker feature than movement. It adds signal at the margins.

### `plate_z`
- **Units:** feet
- **Range:** ~0.5 to 5.0 ft
- **What it captures:** Vertical height as the pitch crosses the plate. The typical strike zone is roughly 1.5–3.5 ft.
- **Why it matters for classification:** Breaking balls (CU, KC) naturally finish below the strike zone. Fastballs finish higher in the zone. Like `plate_x`, this is partly a consequence of pitch type and partly strategic intent.

---

## Pitcher Handedness

### `p_throws`
- **Values:** 0 = Left-handed, 1 = Right-handed
- **What it captures:** Which hand the pitcher throws with.
- **Why it matters for classification:** Handedness is not a pitch characteristic per se, but it is essential **context** for interpreting `pfx_x` and `release_pos_x`. A left-handed pitcher's fastball has the opposite horizontal movement sign from a right-handed pitcher's fastball. Without this feature, the model cannot correctly interpret the direction of movement — it would see the same pitch type appearing on opposite sides of the movement profile.

---

## Feature Summary Table

| Feature | Category | Units | Importance |
|---|---|---|---|
| `pfx_z` | Movement | ft | Very High |
| `release_speed` | Velocity | mph | Very High |
| `pfx_x` | Movement | ft | High |
| `release_spin_rate` | Spin | rpm | High |
| `p_throws` | Context | 0/1 | High |
| `release_pos_x` | Release Point | ft | Moderate |
| `plate_z` | Location | ft | Moderate |
| `release_pos_z` | Release Point | ft | Low–Moderate |
| `release_extension` | Release Point | ft | Low–Moderate |
| `plate_x` | Location | ft | Low |
