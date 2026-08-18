"""
SkyCity Auckland Analytics - Utility & Style Functions
Provides color constants, design tokens, formatters, and reusable UI helpers.
"""

# Color Palette (SkyCity Auckland Branding Theme)
COLOR_PALETTE = {
    'primary': '#1E3A8A',       # Deep Navy / SkyCity Blue
    'secondary': '#0D9488',     # Teal / Cyan accent
    'accent': '#F59E0B',        # Gold / Amber accent
    'dark_bg': '#0F172A',       # Slate 900
    'card_bg': '#1E293B',       # Slate 800
    'text': '#F8FAFC',          # Off-white
    'channels': {
        'In-Store': '#3B82F6',      # Blue (Dine-in / Direct)
        'Uber Eats': '#10B981',     # Emerald / Green (Aggregator)
        'DoorDash': '#EF4444',      # Red (Aggregator)
        'Self-Delivery': '#F59E0B'  # Amber (Direct Digital)
    },
    'dependency': {
        'Low': '#10B981',     # Green (<30%)
        'Medium': '#F59E0B',  # Amber (30-50%)
        'High': '#EF4444'     # Red (>50%)
    }
}

def format_currency(amount: float) -> str:
    """Format float into NZD currency string."""
    if amount is None or amount != amount:  # NaN check
        return "$0.00"
    return f"${amount:,.2f}"

def format_percentage(pct: float) -> str:
    """Format ratio/percentage into formatted string."""
    if pct is None or pct != pct:
        return "0.0%"
    return f"{pct * 100:.1f}%" if pct <= 1.0 else f"{pct:.1f}%"

def format_number(val: int | float) -> str:
    """Format integer or float with thousands separator."""
    if val is None or val != val:
        return "0"
    return f"{val:,.0f}" if isinstance(val, int) or val.is_integer() else f"{val:,.2f}"
