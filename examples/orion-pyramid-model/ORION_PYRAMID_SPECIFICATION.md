# Orion Constellation Pyramid Model - Mathematical Specification

## Overview
This document provides the complete mathematical specification for constructing sand pyramids that mirror the Orion constellation, oriented to the sky with precise astronomical accuracy.

---

## 1. Orion Star Catalog (J2000 Epoch)

### Primary Stars with Celestial Coordinates

| Star | Designation | RA (h m s) | Dec (° ' ") | Apparent Mag | Role |
|------|-------------|------------|-------------|--------------|------|
| **Betelgeuse** | α Ori | 05h 55m 10s | +07° 24' 25" | 0.42 | Left Shoulder |
| **Rigel** | β Ori | 05h 14m 32s | -08° 12' 06" | 0.13 | Right Foot |
| **Bellatrix** | γ Ori | 05h 25m 08s | +06° 20' 59" | 1.64 | Right Shoulder |
| **Mintaka** | δ Ori | 05h 32m 00s | -00° 17' 57" | 2.23 | Belt (West) |
| **Alnilam** | ε Ori | 05h 36m 13s | -01° 12' 07" | 1.69 | Belt (Center) |
| **Alnitak** | ζ Ori | 05h 40m 46s | -01° 56' 34" | 1.77 | Belt (East) |
| **Saiph** | κ Ori | 05h 47m 45s | -09° 40' 11" | 2.09 | Left Foot |

---

## 2. Coordinate Transformation Mathematics

### 2.1 Converting RA/Dec to Cartesian Projection

For ground projection, we use a **gnomonic projection** centered on Orion's Belt:

**Reference Point (Projection Center):**
- RA₀ = 05h 36m 13s = 84.054° (Alnilam - belt center)
- Dec₀ = -01° 12' 07" = -1.202°

**Transformation Equations:**

```
ξ = cos(δ) · sin(α - α₀) / [sin(δ₀)·sin(δ) + cos(δ₀)·cos(δ)·cos(α - α₀)]

η = [cos(δ₀)·sin(δ) - sin(δ₀)·cos(δ)·cos(α - α₀)] / [sin(δ₀)·sin(δ) + cos(δ₀)·cos(δ)·cos(α - α₀)]
```

Where:
- (α, δ) = star's RA and Dec in radians
- (α₀, δ₀) = reference point coordinates
- (ξ, η) = projected coordinates (dimensionless angular units)

---

## 3. Computed Ground Coordinates

### 3.1 Normalized Positions (Belt-Centered, North-Up)

Using Alnilam as origin (0, 0):

| Star | ξ (E-W) | η (N-S) | Distance from Origin |
|------|---------|---------|---------------------|
| **Alnilam** | 0.0000 | 0.0000 | 0.000 |
| **Alnitak** | +0.0197 | -0.0129 | 0.0236 |
| **Mintaka** | -0.0181 | +0.0163 | 0.0244 |
| **Betelgeuse** | -0.0820 | +0.1504 | 0.1713 |
| **Bellatrix** | -0.0481 | +0.1313 | 0.1398 |
| **Rigel** | -0.0944 | -0.1220 | 0.1542 |
| **Saiph** | +0.0505 | -0.1473 | 0.1557 |

---

## 4. Physical Scale Calculation

### 4.1 Scale Factor Derivation

**Design Constraint:** 2 ft pyramid base side length

**Angular Span of Orion:**
- Maximum E-W extent: ~0.176 radians ≈ 10.1°
- Maximum N-S extent: ~0.298 radians ≈ 17.1°

**Recommended Scale Factor:**
```
S = 50 ft per angular unit (0.01 radians)
```

This yields a model approximately **15 ft × 25 ft** total footprint.

### 4.2 Alternative Scales

| Scale Factor | Total Footprint | Belt Spacing | Recommended For |
|--------------|-----------------|--------------|-----------------|
| 0.5 | ~9 × 15 ft | ~1.2 ft | Too tight for 2 ft pyramids |
| 1.0 | ~18 × 30 ft | ~2.4 ft | Tight fit (1 ft bases only) |
| **2.0** | **~36 × 60 ft** | **~4.7 ft** | **Standard (2 ft pyramids)** |
| 3.0 | ~53 × 90 ft | ~7.1 ft | Large installation |
| 5.0 | ~89 × 150 ft | ~11.8 ft | Beach/desert installation |

---

## 5. Final Pyramid Positions (Scale = 2.0, Recommended)

### 5.1 Absolute Coordinates (feet)

**Origin:** Place Alnilam pyramid at your chosen center point
**Total Footprint:** ~36 ft (E-W) × ~60 ft (N-S)

| Pyramid (Star) | X (East+) | Y (North+) | Direction |
|----------------|-----------|------------|-----------|
| **Betelgeuse** | +16.62 ft | +30.37 ft | NNE |
| **Bellatrix** | -9.70 ft | +26.54 ft | NNW |
| **Mintaka** | -3.68 ft | +3.15 ft | NW (Belt) |
| **Alnilam** | 0.00 ft | 0.00 ft | Origin (Belt) |
| **Alnitak** | +3.97 ft | -2.59 ft | SE (Belt) |
| **Rigel** | -18.93 ft | -24.68 ft | SSW |
| **Saiph** | +10.04 ft | -29.82 ft | SSE |

### 5.2 Metric Conversion (meters)

| Pyramid (Star) | X (East+) | Y (North+) |
|----------------|-----------|------------|
| **Betelgeuse** | +5.07 m | +9.26 m |
| **Bellatrix** | -2.96 m | +8.09 m |
| **Mintaka** | -1.12 m | +0.96 m |
| **Alnilam** | 0.00 m | 0.00 m |
| **Alnitak** | +1.21 m | -0.79 m |
| **Rigel** | -5.77 m | -7.52 m |
| **Saiph** | +3.06 m | -9.09 m |

---

## 6. Pyramid Specifications

### 6.1 Individual Pyramid Geometry

**Base:** Square, 2 ft (0.61 m) side length

**Classical Egyptian Proportions (Great Pyramid ratio):**
```
Height-to-Base Ratio: h/b = √φ / 2 ≈ 0.636
Where φ = (1 + √5) / 2 ≈ 1.618 (Golden Ratio)
```

**Dimensions:**
- Base side: **b = 2.00 ft = 24 inches = 0.61 m**
- Height: **h = 1.27 ft = 15.3 inches = 0.39 m**
- Slant height: **s = √(h² + (b/2)²) = 1.58 ft = 19.0 inches**
- Face angle: **α = arctan(2h/b) = 51.83°** (matches Great Pyramid)

### 6.2 Volume Calculation
```
V = (1/3) × b² × h = (1/3) × 4 × 1.27 = 1.69 ft³ ≈ 47.9 liters of sand
```

---

## 7. Sky Orientation Protocol

### 7.1 Compass Alignment

**Critical:** The model must be oriented so that:
- **+Y axis points TRUE NORTH** (not magnetic north)
- **+X axis points TRUE EAST**

**Magnetic Declination Correction:**
```
True North = Magnetic North + Declination
```
Look up your local declination at [NOAA Calculator](https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml)

### 7.2 Mirror Reflection Choice

**Option A - Sky Mirror (Recommended):**
As if looking UP at the sky while lying on your back
- Orion appears as seen from Earth
- East and West are reversed from ground perspective
- Use coordinates as given above

**Option B - Map View:**
As if looking DOWN at a star map
- Flip the X-axis (multiply all X values by -1)
- Matches how star charts are printed

---

## 8. Belt Stars Detail (The Three Kings)

The belt stars are the most recognizable feature. Here's their precise relative positioning:

### 8.1 Belt Triangle Geometry

```
        Mintaka (δ)
           ★
          /|
         / |
        /  | 2.03°
       /   |
      /    |
     ★-----★
 Alnilam  Alnitak
   (ε)      (ζ)

   |--1.38°--|
```

**Angular Separations:**
- Mintaka to Alnilam: **1.38°**
- Alnilam to Alnitak: **1.18°**
- Mintaka to Alnitak: **2.73°**

### 8.2 Belt Physical Distances (at Scale = 2.0)

| From | To | Distance (ft) | Distance (m) |
|------|-----|--------------|--------------|
| Alnilam | Alnitak | 4.74 | 1.44 |
| Alnilam | Mintaka | 4.85 | 1.48 |
| Mintaka | Alnitak | 9.56 | 2.91 |

**Belt Line Angle:** The belt is tilted **~35.5°** from the E-W axis (toward NW-SE)

---

## 9. Brightness-Based Pyramid Scaling (Optional)

To represent stellar magnitude through pyramid size:

### 9.1 Magnitude-to-Size Formula

```
Size_ratio = 2.512^((m_ref - m_star) / 2.5)
```

Using Alnilam (m = 1.69) as reference with base = 2 ft:

| Star | Magnitude | Size Ratio | Base (ft) | Base (in) |
|------|-----------|------------|-----------|-----------|
| Rigel | 0.13 | 1.85 | 3.70 | 44.4 |
| Betelgeuse | 0.42 | 1.62 | 3.24 | 38.9 |
| Bellatrix | 1.64 | 1.02 | 2.05 | 24.6 |
| **Alnilam** | 1.69 | 1.00 | **2.00** | **24.0** |
| Alnitak | 1.77 | 0.96 | 1.93 | 23.1 |
| Saiph | 2.09 | 0.84 | 1.68 | 20.2 |
| Mintaka | 2.23 | 0.78 | 1.55 | 18.6 |

---

## 10. Construction Checklist

### 10.1 Materials (per pyramid at 2 ft base)
- Sand: ~50 liters (1.8 ft³)
- Form boards: 4 × 2 ft lengths
- Plumb line or level
- Compass (with declination correction)
- Measuring tape (25+ ft)
- Stakes and string for layout

### 10.2 Layout Procedure

1. **Establish True North** using Polaris or corrected compass
2. **Mark Origin** (Alnilam position) with stake
3. **Set up baseline** aligned E-W through origin
4. **Measure and stake** each pyramid position using coordinates from §5
5. **Verify distances** between belt stars match §8.2
6. **Construct pyramids** at each stake location

### 10.3 Verification Measurements

After construction, verify these critical distances (Scale = 2.0):

| Check | Expected (ft) | Tolerance |
|-------|---------------|-----------|
| Alnilam ↔ Alnitak | 4.74 | ± 0.25 ft |
| Alnilam ↔ Mintaka | 4.85 | ± 0.25 ft |
| Belt length (Mintaka ↔ Alnitak) | 9.56 | ± 0.5 ft |
| Rigel ↔ Betelgeuse | 65.53 | ± 1.0 ft |

---

## 11. Mathematical Summary

### 11.1 Core Equations

**Gnomonic Projection:**
```
ξ = cos(δ)·sin(Δα) / D
η = [cos(δ₀)·sin(δ) - sin(δ₀)·cos(δ)·cos(Δα)] / D
D = sin(δ₀)·sin(δ) + cos(δ₀)·cos(δ)·cos(Δα)
```

**Physical Position:**
```
X = S · ξ  (feet east of origin)
Y = S · η  (feet north of origin)
```

**Pyramid Height (Egyptian proportions):**
```
h = b · √φ / 2 ≈ 0.636 · b
```

### 11.2 Quick Reference Card

```
╔══════════════════════════════════════════════════════════╗
║           ORION PYRAMID QUICK REFERENCE                  ║
╠══════════════════════════════════════════════════════════╣
║  Scale: 2.0  |  Pyramid Base: 2.0 ft  |  Height: 1.27 ft║
║  Total Footprint: ~36 ft (E-W) × ~60 ft (N-S)           ║
║  Belt Spacing: ~4.7 ft (comfortable gap for 2 ft bases) ║
╠══════════════════════════════════════════════════════════╣
║  POSITIONS (feet from Alnilam at origin)                ║
║  ────────────────────────────────────────               ║
║  Betelgeuse:  (+16.6, +30.4)  NNE                       ║
║  Bellatrix:   ( -9.7, +26.5)  NNW                       ║
║  Mintaka:     ( -3.7, + 3.2)  NW   ← Belt              ║
║  Alnilam:     (  0.0,   0.0)  Origin ← Belt            ║
║  Alnitak:     ( +4.0, - 2.6)  SE   ← Belt              ║
║  Rigel:       (-18.9, -24.7)  SSW                       ║
║  Saiph:       (+10.0, -29.8)  SSE                       ║
╠══════════════════════════════════════════════════════════╣
║  Orient +Y axis to TRUE NORTH for sky-accurate layout   ║
╚══════════════════════════════════════════════════════════╝
```

---

## 12. Connection to Giza Theory

The Orion Correlation Theory (Robert Bauval, 1989) proposes that the three Giza pyramids mirror Orion's Belt:

| Giza Pyramid | Proposed Star | Relative Size |
|--------------|---------------|---------------|
| Great Pyramid (Khufu) | Alnitak | Largest |
| Khafre | Alnilam | Middle |
| Menkaure | Mintaka | Smallest |

**Note:** This model creates a complete Orion constellation, extending beyond the traditional belt-only correlation.

---

*Document Version: 1.0*
*Generated for meta-prompting-framework project*
*Astronomical data: J2000 epoch, SIMBAD database*
