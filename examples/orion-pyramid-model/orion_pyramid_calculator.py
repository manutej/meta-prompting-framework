#!/usr/bin/env python3
"""
Orion Constellation Pyramid Calculator

Computes exact ground positions for sand pyramids that mirror the Orion constellation.
All astronomical coordinates use J2000 epoch data.

Usage:
    python orion_pyramid_calculator.py [--scale SCALE] [--base BASE] [--output FORMAT]

Arguments:
    --scale  : Scale factor in feet per angular unit (default: 50)
    --base   : Pyramid base side length in feet (default: 2.0)
    --output : Output format: 'table', 'json', or 'csv' (default: 'table')
"""

import math
import json
import argparse
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Star:
    """Represents a star with celestial coordinates."""
    name: str
    designation: str
    ra_hours: float      # Right Ascension in decimal hours
    dec_degrees: float   # Declination in decimal degrees
    magnitude: float     # Apparent visual magnitude
    role: str           # Position in constellation

    @property
    def ra_radians(self) -> float:
        """Convert RA to radians."""
        return math.radians(self.ra_hours * 15)  # 15 degrees per hour

    @property
    def dec_radians(self) -> float:
        """Convert Dec to radians."""
        return math.radians(self.dec_degrees)


@dataclass
class PyramidPosition:
    """Computed ground position for a pyramid."""
    star: Star
    xi: float           # Projected E-W coordinate (angular units)
    eta: float          # Projected N-S coordinate (angular units)
    x_feet: float       # Physical X position (feet, East positive)
    y_feet: float       # Physical Y position (feet, North positive)
    x_meters: float     # Physical X position (meters)
    y_meters: float     # Physical Y position (meters)
    base_feet: float    # Pyramid base size (feet)
    height_feet: float  # Pyramid height (feet)


# Orion constellation stars (J2000 epoch coordinates)
ORION_STARS = [
    # Belt stars (The Three Kings)
    Star("Alnilam", "ε Ori", 5.60361, -1.20194, 1.69, "Belt Center"),
    Star("Alnitak", "ζ Ori", 5.67944, -1.94278, 1.77, "Belt East"),
    Star("Mintaka", "δ Ori", 5.53333, -0.29917, 2.23, "Belt West"),

    # Shoulders
    Star("Betelgeuse", "α Ori", 5.91944, 7.40694, 0.42, "Left Shoulder"),
    Star("Bellatrix", "γ Ori", 5.41889, 6.34972, 1.64, "Right Shoulder"),

    # Feet
    Star("Rigel", "β Ori", 5.24222, -8.20167, 0.13, "Right Foot"),
    Star("Saiph", "κ Ori", 5.79583, -9.66972, 2.09, "Left Foot"),
]

# Reference star for projection center (Alnilam - belt center)
REFERENCE_STAR = ORION_STARS[0]

# Golden ratio for Egyptian pyramid proportions
PHI = (1 + math.sqrt(5)) / 2


def gnomonic_projection(star: Star, ref: Star) -> Tuple[float, float]:
    """
    Project celestial coordinates onto a tangent plane using gnomonic projection.

    This projection preserves great circles as straight lines, making it ideal
    for mapping constellation patterns to the ground.

    Args:
        star: Star to project
        ref: Reference star (projection center)

    Returns:
        Tuple of (xi, eta) projected coordinates in angular units
    """
    # Convert to radians
    alpha = star.ra_radians
    delta = star.dec_radians
    alpha_0 = ref.ra_radians
    delta_0 = ref.dec_radians

    # Compute projection denominator
    cos_c = (math.sin(delta_0) * math.sin(delta) +
             math.cos(delta_0) * math.cos(delta) * math.cos(alpha - alpha_0))

    # Gnomonic projection formulas
    xi = math.cos(delta) * math.sin(alpha - alpha_0) / cos_c
    eta = (math.cos(delta_0) * math.sin(delta) -
           math.sin(delta_0) * math.cos(delta) * math.cos(alpha - alpha_0)) / cos_c

    return xi, eta


def calculate_pyramid_height(base: float) -> float:
    """
    Calculate pyramid height using classical Egyptian proportions.

    The Great Pyramid's height-to-base ratio approximates √φ/2 where φ is
    the golden ratio. This produces a face angle of ~51.83°.

    Args:
        base: Base side length

    Returns:
        Pyramid height
    """
    return base * math.sqrt(PHI) / 2


def magnitude_to_size_ratio(mag: float, ref_mag: float = 1.69) -> float:
    """
    Convert stellar magnitude to relative pyramid size.

    Brighter stars (lower magnitude) get larger pyramids.
    Uses the standard magnitude scale where each 2.5 magnitude
    difference corresponds to a factor of 10 in brightness.

    Args:
        mag: Star's apparent magnitude
        ref_mag: Reference magnitude (default: Alnilam's 1.69)

    Returns:
        Size ratio relative to reference
    """
    return math.pow(2.512, (ref_mag - mag) / 2.5)


def calculate_positions(
    scale: float = 50.0,
    base: float = 2.0,
    vary_size: bool = False
) -> List[PyramidPosition]:
    """
    Calculate all pyramid positions and dimensions.

    Args:
        scale: Scale factor (feet per angular unit, where 1 unit ≈ 0.01 radians)
        base: Base pyramid side length in feet
        vary_size: If True, vary pyramid sizes based on stellar magnitude

    Returns:
        List of PyramidPosition objects for each star
    """
    positions = []

    for star in ORION_STARS:
        # Project coordinates
        xi, eta = gnomonic_projection(star, REFERENCE_STAR)

        # Scale to physical units (multiply by 100 to convert angular units)
        x_feet = xi * scale * 100
        y_feet = eta * scale * 100

        # Convert to meters
        x_meters = x_feet * 0.3048
        y_meters = y_feet * 0.3048

        # Calculate pyramid dimensions
        if vary_size:
            size_ratio = magnitude_to_size_ratio(star.magnitude)
            pyramid_base = base * size_ratio
        else:
            pyramid_base = base

        pyramid_height = calculate_pyramid_height(pyramid_base)

        positions.append(PyramidPosition(
            star=star,
            xi=xi,
            eta=eta,
            x_feet=x_feet,
            y_feet=y_feet,
            x_meters=x_meters,
            y_meters=y_meters,
            base_feet=pyramid_base,
            height_feet=pyramid_height
        ))

    return positions


def calculate_distances(positions: List[PyramidPosition]) -> dict:
    """Calculate key distances between pyramids."""
    def dist(p1: PyramidPosition, p2: PyramidPosition) -> float:
        return math.sqrt((p1.x_feet - p2.x_feet)**2 + (p1.y_feet - p2.y_feet)**2)

    # Find specific stars
    by_name = {p.star.name: p for p in positions}

    return {
        "Alnilam_to_Alnitak": dist(by_name["Alnilam"], by_name["Alnitak"]),
        "Alnilam_to_Mintaka": dist(by_name["Alnilam"], by_name["Mintaka"]),
        "Mintaka_to_Alnitak": dist(by_name["Mintaka"], by_name["Alnitak"]),
        "Rigel_to_Betelgeuse": dist(by_name["Rigel"], by_name["Betelgeuse"]),
        "total_span_NS": max(p.y_feet for p in positions) - min(p.y_feet for p in positions),
        "total_span_EW": max(p.x_feet for p in positions) - min(p.x_feet for p in positions),
    }


def print_table(positions: List[PyramidPosition], scale: float, base: float):
    """Print results as formatted table."""
    print("\n" + "="*80)
    print("            ORION CONSTELLATION PYRAMID POSITIONS")
    print("="*80)
    print(f"\n  Scale: {scale} ft per angular unit | Base pyramid: {base} ft")
    print("-"*80)

    # Header
    print(f"{'Star':<12} {'Designation':<8} {'X (ft)':<10} {'Y (ft)':<10} "
          f"{'X (m)':<8} {'Y (m)':<8} {'Base':<6} {'Height':<6}")
    print("-"*80)

    # Sort by Y coordinate (north to south)
    sorted_positions = sorted(positions, key=lambda p: -p.y_feet)

    for p in sorted_positions:
        print(f"{p.star.name:<12} {p.star.designation:<8} "
              f"{p.x_feet:>+9.2f} {p.y_feet:>+9.2f} "
              f"{p.x_meters:>+7.2f} {p.y_meters:>+7.2f} "
              f"{p.base_feet:>5.2f} {p.height_feet:>5.2f}")

    # Distances
    distances = calculate_distances(positions)
    print("\n" + "-"*80)
    print("KEY DISTANCES:")
    print(f"  Belt: Alnilam↔Alnitak = {distances['Alnilam_to_Alnitak']:.2f} ft")
    print(f"  Belt: Alnilam↔Mintaka = {distances['Alnilam_to_Mintaka']:.2f} ft")
    print(f"  Belt: Mintaka↔Alnitak = {distances['Mintaka_to_Alnitak']:.2f} ft (full belt)")
    print(f"  Diagonal: Rigel↔Betelgeuse = {distances['Rigel_to_Betelgeuse']:.2f} ft")
    print(f"\n  Total footprint: {distances['total_span_EW']:.1f} ft (E-W) × "
          f"{distances['total_span_NS']:.1f} ft (N-S)")
    print("="*80)

    # ASCII diagram
    print_ascii_diagram(positions)


def print_ascii_diagram(positions: List[PyramidPosition]):
    """Print ASCII visualization of pyramid layout."""
    print("\n" + "="*60)
    print("         LAYOUT DIAGRAM (North ↑, East →)")
    print("="*60 + "\n")

    # Create grid
    width, height = 50, 30
    grid = [[' ' for _ in range(width)] for _ in range(height)]

    # Find bounds
    min_x = min(p.x_feet for p in positions)
    max_x = max(p.x_feet for p in positions)
    min_y = min(p.y_feet for p in positions)
    max_y = max(p.y_feet for p in positions)

    # Map positions to grid
    for p in positions:
        # Normalize to grid coordinates
        gx = int((p.x_feet - min_x) / (max_x - min_x + 0.01) * (width - 6)) + 3
        gy = int((max_y - p.y_feet) / (max_y - min_y + 0.01) * (height - 4)) + 2

        # Place marker
        gx = max(1, min(width-2, gx))
        gy = max(0, min(height-1, gy))

        # Use first letter of star name
        marker = p.star.name[0]
        if p.star.name in ["Alnilam", "Alnitak"]:
            marker = p.star.name[:2]
        grid[gy][gx] = marker

    # Print grid with border
    print("    +" + "-"*(width) + "+")
    print("  N |" + " "*((width-5)//2) + "NORTH" + " "*((width-5)//2) + "|")
    for row in grid:
        print("    |" + "".join(row) + "|")
    print("    +" + "-"*(width) + "+")
    print("     W" + " "*(width//2-3) + "E")

    print("""
    Legend: B=Betelgeuse  Be=Bellatrix  M=Mintaka
            Al=Alnilam    An=Alnitak    R=Rigel  S=Saiph
    """)


def output_json(positions: List[PyramidPosition], scale: float, base: float):
    """Output results as JSON."""
    data = {
        "parameters": {
            "scale_ft_per_unit": scale,
            "base_pyramid_ft": base,
            "height_to_base_ratio": math.sqrt(PHI) / 2
        },
        "pyramids": [
            {
                "star": p.star.name,
                "designation": p.star.designation,
                "magnitude": p.star.magnitude,
                "role": p.star.role,
                "position": {
                    "x_feet": round(p.x_feet, 2),
                    "y_feet": round(p.y_feet, 2),
                    "x_meters": round(p.x_meters, 2),
                    "y_meters": round(p.y_meters, 2)
                },
                "dimensions": {
                    "base_feet": round(p.base_feet, 2),
                    "height_feet": round(p.height_feet, 2)
                }
            }
            for p in positions
        ],
        "distances": calculate_distances(positions)
    }
    print(json.dumps(data, indent=2))


def output_csv(positions: List[PyramidPosition]):
    """Output results as CSV."""
    print("Star,Designation,Magnitude,X_feet,Y_feet,X_meters,Y_meters,Base_feet,Height_feet")
    for p in positions:
        print(f"{p.star.name},{p.star.designation},{p.star.magnitude},"
              f"{p.x_feet:.2f},{p.y_feet:.2f},{p.x_meters:.2f},{p.y_meters:.2f},"
              f"{p.base_feet:.2f},{p.height_feet:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate Orion constellation pyramid positions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python orion_pyramid_calculator.py
    python orion_pyramid_calculator.py --scale 100 --base 1.5
    python orion_pyramid_calculator.py --output json > pyramids.json
        """
    )
    parser.add_argument("--scale", type=float, default=50.0,
                        help="Scale factor in feet per angular unit (default: 50)")
    parser.add_argument("--base", type=float, default=2.0,
                        help="Pyramid base side length in feet (default: 2.0)")
    parser.add_argument("--output", choices=["table", "json", "csv"], default="table",
                        help="Output format (default: table)")
    parser.add_argument("--vary-size", action="store_true",
                        help="Vary pyramid sizes based on stellar magnitude")

    args = parser.parse_args()

    positions = calculate_positions(args.scale, args.base, args.vary_size)

    if args.output == "table":
        print_table(positions, args.scale, args.base)
    elif args.output == "json":
        output_json(positions, args.scale, args.base)
    elif args.output == "csv":
        output_csv(positions)


if __name__ == "__main__":
    main()
