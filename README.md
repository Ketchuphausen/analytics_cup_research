# Off-Ball Run Analysis: Quantifying Space Creation in Football

**Author:** Ivo Steinke  
**Competition:** PySport X SkillCorner Analytics Cup

---

## Research Track Abstract

### Introduction

Off-ball movement is crucial to modern football tactics, yet traditional analysis focuses primarily on players in possession. This work presents a novel approach to quantify how high-speed off-ball runs create exploitable space for the ball carrier, addressing the question: *How much space do coordinated off-ball movements generate during team possession?*

The methodology analyzes 10 A-League matches using SkillCorner's broadcast tracking data, combining velocity-based run detection with Voronoi tessellation to measure spatial impact. This transparency-first approach provides interpretable metrics for tactical analysis without relying on black-box machine learning.

### Methods

**Data:** The analysis covers 10 A-League matches (2024/25 season) containing 688,000+ player position frames at 10 Hz, focusing on runs during own-team possession phases.

**Run Detection:** High-speed off-ball movements were identified using three criteria:
- Velocity threshold: ≥5.0 m/s (18 km/h)
- Minimum duration: 3 seconds sustained movement
- Context filter: Own team in possession, runner without ball

**Space Measurement:** For each detected run, Voronoi diagrams were calculated at run start (t₀) and end (t₀+3s) to quantify controlled area changes. Space creation was measured specifically for the ball carrier:
```
Space_Created = Voronoi_Area(ball_carrier, t₀+3s) - Voronoi_Area(ball_carrier, t₀)
```

**Normalization:** All runs were normalized to attack direction (left-to-right) accounting for teams switching sides at halftime, enabling cross-match spatial pattern analysis.

![Figure 1: Voronoi Methodology](figs/voronoi_example.png)

*Figure 1: Voronoi tessellation showing controlled space per player. The ball carrier (gold square) gains exploitable space when teammates make off-ball runs, pulling defenders away.*

### Results

Across 10 matches, **523 distinct off-ball runs** were detected (average 52.3 runs/match) with an average of **5,692 m²** of space created per run for ball carriers.

**Key Findings:**
- Average space creation: **5,692 m²** per run (~80% of pitch area)
- Average run velocity: 6.49 m/s (23.3 km/h)
- Average run duration: 0.9 seconds

**Top Performers:**
1. **Most Runs/Match:** Thomas Aquilina (Newcastle United Jets FC) - 8.0 runs/match
2. **Most Space/Match:** Kai Trewin (Melbourne City FC) - 94,171 m²/match
3. **Most Efficient:** Jordi Valadon (Melbourne Victory FC) - 25,317 m²/run (min. 5 runs)

![Figure 2: Trajectory Patterns](figs/trajectory_visualization.png)

*Figure 2: All off-ball runs from Melbourne Victory (away team) showing normalized trajectories. Green markers indicate run start, red markers show run end. Arrows reveal coordinated movement patterns creating space in attacking third.*

### Conclusion

This transparent, Voronoi-based approach successfully quantifies off-ball run effectiveness using broadcast tracking data. The methodology reveals measurable spatial advantages created by coordinated off-ball movement, with runs creating average exploitable space equivalent to 80% of pitch area for ball carriers.

**Limitations:** Broadcast tracking captures only 51% of on-pitch moments due to camera coverage constraints. Future work should incorporate GPS data for complete coverage and extend analysis to pressing scenarios (runs during opponent possession).

**Impact:** This interpretable framework enables coaches to evaluate off-ball movement patterns quantitatively, informing tactical preparation and player development without requiring complex machine learning infrastructure.

---

## Quick Start
```python
from src.data_loader import load_matches_info, load_match_data
from src.space_analysis import analyze_all_matches_normalized

# Define match IDs
MATCH_IDS = ['2017461', '1996435', '1886347', '1899585', '1925299',
             '1953632', '2006229', '2011166', '2013725', '2015213']

# Load and analyze
matches = load_matches_info(MATCH_IDS)
trajectories = analyze_all_matches_normalized(matches, load_match_data, velocity_threshold=5.0)

print(f"Detected {len(trajectories)} runs")
print(f"Average space created: {trajectories['total_space_created'].mean():.0f} m²")
```

## License

MIT License - See LICENSE file for details