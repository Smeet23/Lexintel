"""Default theme configuration for new firms.

Includes standard shadcn/ui tokens AND Lexintel custom tokens
(surface, surface-hover, border-strong, primary-light, accent-hover,
accent-muted, success, success-light, warning, warning-light).
"""

DEFAULT_THEME_CONFIG = {
    "light": {
        # Standard shadcn/ui tokens
        "background": "#FAFAF8",
        "foreground": "#111111",
        "primary": "#111111",
        "primary-foreground": "#FAFAF8",
        "secondary": "#F5F5F3",
        "secondary-foreground": "#111111",
        "accent": "#111111",
        "accent-foreground": "#FFFFFF",
        "muted": "#F5F5F3",
        "muted-foreground": "#999999",
        "destructive": "#DC2626",
        "destructive-foreground": "#FFFFFF",
        "card": "#FFFFFF",
        "card-foreground": "#111111",
        "popover": "#FFFFFF",
        "popover-foreground": "#111111",
        "border": "#E5E4E2",
        "input": "#E5E4E2",
        "ring": "#111111",
        "chart-1": "#111111",
        "chart-2": "#444444",
        "chart-3": "#777777",
        "chart-4": "#AAAAAA",
        "chart-5": "#DDDDDD",
        "sidebar-background": "#F5F4F2",
        "sidebar-foreground": "#111111",
        "sidebar-accent": "#FFFFFF",
        "sidebar-border": "#E5E4E2",
        # Lexintel custom tokens (used 168+ times across codebase)
        "primary-light": "#222222",
        "accent-hover": "#2A2A2A",
        "accent-muted": "#F5F5F3",
        "surface": "#F5F4F2",
        "surface-hover": "#EDECEB",
        "border-strong": "#D1D0CE",
        "success": "#16A34A",
        "success-light": "#F0FDF4",
        "warning": "#CA8A04",
        "warning-light": "#FEFCE8",
    },
    "dark": {
        # Standard shadcn/ui tokens
        "background": "#0A0A0A",
        "foreground": "#FAFAF8",
        "primary": "#FAFAF8",
        "primary-foreground": "#111111",
        "secondary": "#1A1A1A",
        "secondary-foreground": "#FAFAF8",
        "accent": "#FAFAF8",
        "accent-foreground": "#111111",
        "muted": "#1A1A1A",
        "muted-foreground": "#666666",
        "destructive": "#EF4444",
        "destructive-foreground": "#FFFFFF",
        "card": "#141414",
        "card-foreground": "#FAFAF8",
        "popover": "#141414",
        "popover-foreground": "#FAFAF8",
        "border": "#2A2A2A",
        "input": "#2A2A2A",
        "ring": "#FAFAF8",
        "chart-1": "#FAFAF8",
        "chart-2": "#CCCCCC",
        "chart-3": "#999999",
        "chart-4": "#666666",
        "chart-5": "#333333",
        "sidebar-background": "#0F0F0F",
        "sidebar-foreground": "#FAFAF8",
        "sidebar-accent": "#1A1A1A",
        "sidebar-border": "#2A2A2A",
        # Lexintel custom tokens
        "primary-light": "#DDDDDD",
        "accent-hover": "#CCCCCC",
        "accent-muted": "#1A1A1A",
        "surface": "#141414",
        "surface-hover": "#1E1E1E",
        "border-strong": "#3A3A3A",
        "success": "#22C55E",
        "success-light": "#052E16",
        "warning": "#EAB308",
        "warning-light": "#422006",
    },
    "typography": {
        "font-sans": "DM Sans",
        "font-display": "DM Serif Display",
        "font-mono": "JetBrains Mono",
    },
    "layout": {
        "radius-sm": "0.375rem",
        "radius-md": "0.5rem",
        "radius-lg": "0.75rem",
        "radius-xl": "1rem",
    },
    "shadows": {
        "color": "#000000",
        "opacity": "0.04",
        "blur": "12",
        "spread": "0",
        "offset-x": "0",
        "offset-y": "4",
    },
}

# All valid theme token keys for validation
VALID_COLOR_TOKENS = list(DEFAULT_THEME_CONFIG["light"].keys())
VALID_TYPOGRAPHY_TOKENS = list(DEFAULT_THEME_CONFIG["typography"].keys())
VALID_LAYOUT_TOKENS = list(DEFAULT_THEME_CONFIG["layout"].keys())
VALID_SHADOW_TOKENS = list(DEFAULT_THEME_CONFIG["shadows"].keys())
