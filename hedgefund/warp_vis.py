#%%
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
# Set plotting style
sns.set(style="white", context="talk")
plt.rcParams['savefig.facecolor'] = 'white'



def draw_mlp_trapezoid(ax, center, width, height_in, height_out, label, color='#d2b48c'):
    """Draws the specific MLP trapezoid shape."""
    x, y = center
    # Define vertices: (bottom-left, top-left, top-right, bottom-right)
    verts = [
        (x - width/2, y - height_in/2),
        (x - width/2, y + height_in/2),
        (x + width/2, y + height_out/2),
        (x + width/2, y - height_out/2)
    ]
    poly = patches.Polygon(verts, closed=True, facecolor=color, edgecolor='gray', alpha=0.9, zorder=2)
    ax.add_patch(poly)
    ax.text(x, y, label, ha='center', va='center', fontsize=12, fontweight='bold', zorder=3)
    return x + width/2, x - width/2  # return right and left x-coords

def draw_block(ax, center, width, height, text, color, fontsize=12, text_color='black'):
    """Draws a rectangular block with text."""
    x, y = center
    rect = patches.Rectangle((x - width/2, y - height/2), width, height, 
                             facecolor=color, edgecolor='none', alpha=1.0, zorder=2)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, color=text_color, zorder=3)
    return rect

def draw_circle(ax, center, radius, text, color='#e0e0e0'):
    """Draws a circular node (for the addition operation)."""
    x, y = center
    circle = patches.Circle((x, y), radius, facecolor=color, edgecolor='gray', zorder=2)
    ax.add_patch(circle)
    ax.text(x, y, text, ha='center', va='center', fontsize=14, fontweight='bold', zorder=3)

def connect(ax, xy_from, xy_to, color='black', style='->', linestyle='solid', connectionstyle=None):
    """Connects two points with an arrow."""
    if connectionstyle:
        arrow = patches.FancyArrowPatch(posA=xy_from, posB=xy_to, 
                                        arrowstyle=style, color=color, 
                                        linestyle=linestyle, connectionstyle=connectionstyle,
                                        mutation_scale=15, zorder=1)
        ax.add_patch(arrow)
    else:
        ax.annotate("", xy=xy_to, xytext=xy_from, 
                    arrowprops=dict(arrowstyle=style, color=color, linestyle=linestyle, lw=1.5))

# --- Setup Figure ---
fig, axes = plt.subplots(2, 1, figsize=(10, 12))
plt.subplots_adjust(hspace=0.4)

# Colors matching the reference
c_mlp = '#cfb086'      # Tan/Beige
c_node = '#8ab6b9'     # Teal/Blue
c_loss = '#ff6b6b'     # Red
c_true = '#cda5a5'     # Pinkish Brown
c_sample = '#d8bfd8'   # Light Purple
c_bg = 'white'

# ==========================================
# 1. Deterministic Mode (Top Plot)
# ==========================================
ax = axes[0]
ax.set_title("Deterministic", loc='left', fontsize=28, fontweight='bold', y=0.85)
ax.set_xlim(-2, 12)
ax.set_ylim(-4, 4)
ax.axis('off')

# -- Nodes --
# Input Text
ax.text(-1, 0, r'$\tau$', fontsize=18, ha='center', va='center')

# MLP
mlp_x, mlp_left = draw_mlp_trapezoid(ax, (2, 0), width=3, height_in=3, height_out=5, label=r'$\bf{MLP}_{\theta_t}$', color=c_mlp)

# y_hat block
draw_block(ax, (6, 0), 2, 1.5, r'$\hat{y}_t$', c_node)

# MSE Loss block
draw_block(ax, (10, 0), 2, 1.5, 'MSE\nLoss', c_loss, fontsize=14, text_color='black')

# y_true block
draw_block(ax, (10, -3), 2, 1.5, r'$y^{true}_t$', c_true)

# -- Connections --
# Input -> MLP
connect(ax, (-0.5, 0), (mlp_left, 0))

# MLP -> y_hat
connect(ax, (mlp_x, 0), (5, 0))

# y_hat -> Loss
connect(ax, (7, 0), (9, 0))

# y_true -> Loss
connect(ax, (10, -2.25), (10, -0.75))

# ==========================================
# 2. Stochastic Mode (Bottom Plot)
# ==========================================
ax = axes[1]
ax.set_title("Stochastic", loc='left', fontsize=28, fontweight='bold', y=0.90)
ax.set_xlim(-2, 12)
ax.set_ylim(-5, 5)
ax.axis('off')

# -- Nodes --
# Input Text
ax.text(-1, 0, r'$\tau$', fontsize=18, ha='center', va='center')

# MLP
mlp_x, mlp_left = draw_mlp_trapezoid(ax, (2, 0), width=3, height_in=3, height_out=5, label=r'$\bf{MLP}_{\theta_t}$', color=c_mlp)

# mu block
draw_block(ax, (6, 2), 1.8, 1.2, r'$\mu_t$', c_node)

# sigma block
draw_block(ax, (6, -2), 1.8, 1.2, r'$\sigma_t$', c_node)

# Addition Circle
draw_circle(ax, (8, -0.5), 0.6, '+')

# Sample block
draw_block(ax, (9.5, -0.5), 1, 1.0, r'$\hat{y}_t$', c_sample)
ax.text(9.5, -1.5, "(For Next Step Input\nDifference)", fontsize=8, color='gray', ha='center')

# NLL Loss block
draw_block(ax, (10.5, 3), 1.8, 1.5, 'NLL\nLoss', c_loss, fontsize=14)

# y_true block
draw_block(ax, (10.5, -4), 1.8, 1.5, r'$y^{true}_t$', c_true)

# Epsilon Text
ax.text(8, -3, r'$\epsilon \sim \mathcal{N}(0, I)$', fontsize=12, color='gray', ha='center')

# -- Connections --
# Input -> MLP
connect(ax, (-0.5, 0), (mlp_left, 0))

# MLP -> mu (angled)
connect(ax, (mlp_x, 0.5), (5.1, 2))

# MLP -> sigma (angled)
connect(ax, (mlp_x, -0.5), (5.1, -2))

# mu -> +
connect(ax, (6.9, 2), (7.7, -0.1))

# sigma -> +
connect(ax, (6.9, -2), (7.7, -0.9))

# epsilon -> +
connect(ax, (8, -2.5), (8, -1.1))

# + -> sample
connect(ax, (8.6, -0.5), (9.0, -0.5))

# y_true -> Loss
connect(ax, (10.5, -3.25), (10.5, 2.25))

# -- Dashed Red Connections (Loss Input) --
# mu -> Loss
connect(ax, (6.9, 2), (9.6, 3), color='#eb6b6b', linestyle='dashed', connectionstyle="arc3,rad=-0.1")

# sigma -> Loss
connect(ax, (6.9, -2), (9.6, 2.8), color='#eb6b6b', linestyle='dashed', connectionstyle="arc3,rad=-0.3")

plt.tight_layout()
plt.show()