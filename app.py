"""
app.py  –  Streamlit Dashboard for PSO Lab 9
Pastel-themed interactive visualisation.

Run:
    streamlit run app.py
"""

import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from pso import (
    rosenbrock_constrained,
    gbest_pso,
    lbest_pso,
    compare_algorithms,
    PSOResult,
)

# ──────────────────────────────────────────────────────────
#  Page config & global pastel palette
# ──────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PSO Lab 9 Dashboard",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

PASTEL = {
    "lavender":   "#C9B8E8",
    "mint":       "#B8E8D0",
    "peach":      "#F9CBA7",
    "sky":        "#A8D8EA",
    "blush":      "#F7C5CC",
    "lemon":      "#FFF0A5",
    "lilac_bg":   "#F5F0FF",
    "card_bg":    "#FFFFFF",
    "text":       "#3D3055",
    "subtext":    "#7A6E9A",
    "border":     "#E0D9F5",
    "gbest_line": "#9B72CF",
    "lbest_line": "#5DAA7F",
    "particle":   "#F4A4B0",
    "pbest":      "#A4C4F4",
    "gbest_dot":  "#FFD166",
}

# ──────────────────────────────────────────────────────────
#  CSS injection
# ──────────────────────────────────────────────────────────

st.markdown(f"""
<style>
  /* Global font & background */
  html, body, [class*="css"] {{
      font-family: 'Segoe UI', sans-serif;
      background-color: {PASTEL['lilac_bg']};
      color: {PASTEL['text']};
  }}

  /* Sidebar */
  [data-testid="stSidebar"] {{
      background: linear-gradient(160deg, #EDE7FF 0%, #D8F3E8 100%);
      border-right: 1px solid {PASTEL['border']};
  }}
  [data-testid="stSidebar"] * {{
      color: {PASTEL['text']} !important;
  }}

  /* Header strip */
  .page-header {{
      background: linear-gradient(120deg, {PASTEL['lavender']}, {PASTEL['sky']}, {PASTEL['mint']});
      padding: 1.4rem 2rem;
      border-radius: 16px;
      margin-bottom: 1.2rem;
      box-shadow: 0 4px 16px rgba(155,114,207,0.15);
  }}
  .page-header h1 {{
      color: {PASTEL['text']};
      font-size: 1.9rem;
      margin: 0;
  }}
  .page-header p {{
      color: {PASTEL['subtext']};
      margin: 0.3rem 0 0;
  }}

  /* Metric cards */
  .metric-card {{
      background: {PASTEL['card_bg']};
      border: 1px solid {PASTEL['border']};
      border-radius: 14px;
      padding: 1rem 1.2rem;
      text-align: center;
      box-shadow: 0 2px 10px rgba(155,114,207,0.08);
  }}
  .metric-card .val {{
      font-size: 1.6rem;
      font-weight: 700;
      color: {PASTEL['text']};
  }}
  .metric-card .lbl {{
      font-size: 0.78rem;
      color: {PASTEL['subtext']};
      margin-top: 0.2rem;
  }}

  /* Section headers */
  .section-title {{
      font-size: 1.1rem;
      font-weight: 700;
      color: {PASTEL['text']};
      border-left: 4px solid {PASTEL['lavender']};
      padding-left: 0.6rem;
      margin: 1.2rem 0 0.6rem;
  }}

  /* Badge pills */
  .badge {{
      display: inline-block;
      padding: 0.2rem 0.7rem;
      border-radius: 20px;
      font-size: 0.78rem;
      font-weight: 600;
  }}
  .badge-gbest {{ background:{PASTEL['lavender']}; color:{PASTEL['text']}; }}
  .badge-lbest {{ background:{PASTEL['mint']};     color:{PASTEL['text']}; }}

  /* Tab strip */
  .stTabs [data-baseweb="tab-list"] {{
      gap: 6px;
      background: transparent;
  }}
  .stTabs [data-baseweb="tab"] {{
      background: #EDE7FF;
      border-radius: 10px 10px 0 0;
      color: {PASTEL['text']};
      font-weight: 600;
      padding: 0.5rem 1.2rem;
  }}
  .stTabs [aria-selected="true"] {{
      background: {PASTEL['lavender']} !important;
      color: {PASTEL['text']} !important;
  }}

  /* Buttons */
  .stButton > button {{
      background: linear-gradient(135deg, {PASTEL['lavender']}, {PASTEL['sky']});
      color: {PASTEL['text']};
      border: none;
      border-radius: 10px;
      font-weight: 600;
      padding: 0.5rem 1.5rem;
      width: 100%;
  }}
  .stButton > button:hover {{
      opacity: 0.88;
      transform: translateY(-1px);
  }}

  /* Sliders & inputs */
  .stSlider > div > div > div > div {{ background: {PASTEL['lavender']}; }}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
  <h1>🌸 Particle Swarm Optimization – Lab 9</h1>
  <p>Evolutionary Computing </p>
</div>
""", unsafe_allow_html=True)
# ──────────────────────────────────────────────────────────
#  Sidebar controls
# ──────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ PSO Parameters")
    st.markdown("---")

    algo_choice = st.radio(
        "Algorithm (Task 1)",
        ["gbest PSO", "lbest PSO", "Both (Compare)"],
        index=0,
    )

    st.markdown("#### 🐦 Swarm Settings")
    n_particles   = st.slider("Number of Particles",   10, 100, 30, 5)
    n_iterations  = st.slider("Iterations",            50, 500, 200, 10)

    st.markdown("#### 🎛️ Velocity Coefficients")
    w   = st.slider("Inertia Weight (w)",       0.1, 1.0, 0.7, 0.05)
    c1  = st.slider("Cognitive Coeff (c₁)",     0.5, 3.0, 1.5, 0.1)
    c2  = st.slider("Social Coeff (c₂)",        0.5, 3.0, 1.5, 0.1)

    if algo_choice in ["lbest PSO", "Both (Compare)"]:
        st.markdown("#### 🔗 Ring Topology")
        nbr_size = st.slider("Neighbourhood Size", 2, 10, 3, 1)
    else:
        nbr_size = 3

    if algo_choice == "Both (Compare)":
        st.markdown("#### 📊 Comparison Runs")
        n_runs = st.slider("# Independent Runs", 2, 20, 5, 1)
    else:
        n_runs = 5

    st.markdown("#### 🌱 Reproducibility")
    seed = st.number_input("Random Seed", 0, 9999, 42, 1)

    st.markdown("---")
    run_btn = st.button("🚀 Run PSO", use_container_width=True)

BOUNDS = [(-5.0, 5.0), (-5.0, 5.0)]

# ──────────────────────────────────────────────────────────
#  Helper: plotly theme defaults
# ──────────────────────────────────────────────────────────

def _layout(fig, title="", height=420):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=PASTEL["text"])),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.6)",
        font=dict(color=PASTEL["text"], size=12),
        height=height,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor=PASTEL["border"],
            borderwidth=1,
            font=dict(size=11),
        ),
    )
    fig.update_xaxes(gridcolor=PASTEL["border"], zeroline=False)
    fig.update_yaxes(gridcolor=PASTEL["border"], zeroline=False)
    return fig


# ──────────────────────────────────────────────────────────
#  Plots
# ──────────────────────────────────────────────────────────

def plot_convergence(results: list[PSOResult]) -> go.Figure:
    colors = [PASTEL["gbest_line"], PASTEL["lbest_line"],
              PASTEL["peach"], PASTEL["blush"]]
    fig = go.Figure()
    for i, r in enumerate(results):
        fig.add_trace(go.Scatter(
            y=r.convergence,
            mode="lines",
            name=r.algorithm,
            line=dict(color=colors[i % len(colors)], width=2.5),
        ))
    fig.update_yaxes(type="log", title="Best Fitness (log scale)")
    fig.update_xaxes(title="Iteration")
    return _layout(fig, "📉 Convergence Curve", height=380)


def plot_swarm_snapshot(result: PSOResult, iteration: int) -> go.Figure:
    """2D scatter of particles at a given iteration."""
    if not result.all_positions or iteration >= len(result.all_positions):
        return go.Figure()

    pos = result.all_positions[iteration]  # shape (n_particles, 2)

    # Contour of the objective function
    grid_n = 60
    x1 = np.linspace(-5, 5, grid_n)
    x2 = np.linspace(-5, 5, grid_n)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.vectorize(lambda a, b: rosenbrock_constrained(np.array([a, b])))(X1, X2)

    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=x1, y=x2, z=np.log1p(Z),
        colorscale=[
            [0,   "#EDE7FF"],
            [0.3, "#C9B8E8"],
            [0.6, "#A8D8EA"],
            [1,   "#5D4A8A"],
        ],
        showscale=False,
        contours=dict(coloring="heatmap"),
        opacity=0.55,
    ))
    fig.add_trace(go.Scatter(
        x=pos[:, 0], y=pos[:, 1],
        mode="markers",
        name="Particles",
        marker=dict(color=PASTEL["particle"], size=9,
                    line=dict(color="#D6607A", width=1)),
    ))
    # Global best marker (star)
    fig.add_trace(go.Scatter(
        x=[result.best_position[0]],
        y=[result.best_position[1]],
        mode="markers",
        name="Global Best",
        marker=dict(symbol="star", color=PASTEL["gbest_dot"],
                    size=18, line=dict(color="#C8920A", width=1.5)),
    ))
    # Known optimum
    fig.add_trace(go.Scatter(
        x=[1.0], y=[1.0],
        mode="markers",
        name="True Optimum (1,1)",
        marker=dict(symbol="cross", color="#60B06E", size=14,
                    line=dict(color="#2D7A3F", width=2)),
    ))
    fig.update_xaxes(range=[-5.2, 5.2], title="x₁")
    fig.update_yaxes(range=[-5.2, 5.2], title="x₂")
    return _layout(fig, f"🔍 Swarm at Iteration {iteration + 1}", height=400)


def plot_comparison_bar(cmp: dict) -> go.Figure:
    """Side-by-side bar for mean fitness and time."""
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Mean Best Fitness (lower=better)",
                                        "Mean Runtime (ms)"])
    algos  = ["gbest PSO", "lbest PSO"]
    colors = [PASTEL["lavender"], PASTEL["mint"]]
    fits   = [cmp["gbest_stats"]["best_fitness_mean"],
              cmp["lbest_stats"]["best_fitness_mean"]]
    fit_e  = [cmp["gbest_stats"]["best_fitness_std"],
              cmp["lbest_stats"]["best_fitness_std"]]
    times  = [cmp["gbest_stats"]["time_mean"] * 1000,
              cmp["lbest_stats"]["time_mean"] * 1000]
    time_e = [cmp["gbest_stats"]["time_std"] * 1000,
              cmp["lbest_stats"]["time_std"] * 1000]

    for i, (a, c) in enumerate(zip(algos, colors)):
        fig.add_trace(go.Bar(
            x=[a], y=[fits[i]],
            error_y=dict(type="data", array=[fit_e[i]], visible=True),
            name=a, marker_color=c, showlegend=(i == 0),
            width=0.45,
        ), row=1, col=1)
        fig.add_trace(go.Bar(
            x=[a], y=[times[i]],
            error_y=dict(type="data", array=[time_e[i]], visible=True),
            name=a, marker_color=c, showlegend=False,
            width=0.45,
        ), row=1, col=2)

    fig.update_yaxes(type="log", row=1, col=1)
    return _layout(fig, "📊 Task 2 – Algorithm Comparison", height=380)


def plot_multi_run_box(cmp: dict) -> go.Figure:
    """Box plots of fitness distribution across runs."""
    gbest_fits = [r.best_fitness for r in cmp["gbest"]]
    lbest_fits = [r.best_fitness for r in cmp["lbest"]]

    fig = go.Figure()
    fig.add_trace(go.Box(
        y=gbest_fits, name="gbest PSO",
        marker_color=PASTEL["lavender"],
        line_color=PASTEL["gbest_line"],
        boxmean="sd",
    ))
    fig.add_trace(go.Box(
        y=lbest_fits, name="lbest PSO",
        marker_color=PASTEL["mint"],
        line_color=PASTEL["lbest_line"],
        boxmean="sd",
    ))
    fig.update_yaxes(type="log", title="Best Fitness (log)")
    return _layout(fig, "🎻 Fitness Distribution across Runs", height=360)


def plot_landscape() -> go.Figure:
    """3-D surface of the objective function."""
    grid_n = 55
    x1 = np.linspace(-5, 5, grid_n)
    x2 = np.linspace(-5, 5, grid_n)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.vectorize(lambda a, b: rosenbrock_constrained(np.array([a, b])))(X1, X2)

    fig = go.Figure(go.Surface(
        x=X1, y=X2, z=np.log1p(Z),
        colorscale=[
            [0,   "#EDE7FF"],
            [0.2, "#C9B8E8"],
            [0.5, "#A8D8EA"],
            [0.8, "#72B5D8"],
            [1,   "#3D5A80"],
        ],
        opacity=0.88,
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor="#FFD166", project_z=True)
        ),
    ))
    fig.update_layout(
        scene=dict(
            xaxis_title="x₁",
            yaxis_title="x₂",
            zaxis_title="log(1+f)",
            bgcolor="rgba(245,240,255,0.4)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PASTEL["text"]),
        title=dict(
            text="🏔️ Fitness Landscape  [log scale]",
            font=dict(size=14, color=PASTEL["text"]),
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        height=430,
    )
    return fig


# ──────────────────────────────────────────────────────────
#  Main content
# ──────────────────────────────────────────────────────────

tab_task1, tab_task2 = st.tabs(
    ["🎯 Task 1 – Minimisation", "📊 Task 2 – Comparison"]
)

# ── Session state ──
if "results" not in st.session_state:
    st.session_state["results"]    = None
    st.session_state["cmp"]        = None
    st.session_state["algo_choice"] = None

# ── Run ──
if run_btn:
    with st.spinner("🌀 Running PSO…"):
        kwargs = dict(
            func=rosenbrock_constrained,
            bounds=BOUNDS,
            n_particles=n_particles,
            n_iterations=n_iterations,
            w=w, c1=c1, c2=c2,
        )
        if algo_choice == "gbest PSO":
            r = gbest_pso(**kwargs, seed=int(seed))
            st.session_state["results"] = [r]
            st.session_state["cmp"] = None
        elif algo_choice == "lbest PSO":
            r = lbest_pso(**kwargs, neighborhood_size=nbr_size, seed=int(seed))
            st.session_state["results"] = [r]
            st.session_state["cmp"] = None
        else:  # Both
            rg = gbest_pso(**kwargs, seed=int(seed))
            rl = lbest_pso(**kwargs, neighborhood_size=nbr_size, seed=int(seed))
            st.session_state["results"] = [rg, rl]
            cmp = compare_algorithms(
                **kwargs,
                neighborhood_size=nbr_size,
                seed=int(seed),
                n_runs=n_runs,
            )
            st.session_state["cmp"] = cmp
        st.session_state["algo_choice"] = algo_choice

# ── Task 1 tab ──
with tab_task1:
    results = st.session_state.get("results")
    if results is None:
        st.info("👈 Set your parameters in the sidebar and click **Run PSO**.")
    else:
        # ---- Metric row ----
        r0 = results[0]
        st.markdown('<div class="section-title">Results</div>',
                    unsafe_allow_html=True)
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.markdown(f"""<div class="metric-card">
                <div class="val">{r0.best_fitness:.6f}</div>
                <div class="lbl">Best Fitness</div></div>""",
                unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="metric-card">
                <div class="val">{r0.best_position[0]:.4f}</div>
                <div class="lbl">x₁</div></div>""",
                unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class="metric-card">
                <div class="val">{r0.best_position[1]:.4f}</div>
                <div class="lbl">x₂</div></div>""",
                unsafe_allow_html=True)
        with m4:
            st.markdown(f"""<div class="metric-card">
                <div class="val">{r0.elapsed_time*1000:.1f} ms</div>
                <div class="lbl">Runtime</div></div>""",
                unsafe_allow_html=True)
        with m5:
            st.markdown(f"""<div class="metric-card">
                <div class="val">{r0.evaluations:,}</div>
                <div class="lbl">Evaluations</div></div>""",
                unsafe_allow_html=True)

        st.markdown("")

        # ---- Convergence ----
        st.plotly_chart(plot_convergence(results), use_container_width=True, key="convergence_task1")

        # ---- Swarm animation slider ----
        if r0.all_positions:
            st.markdown('<div class="section-title">🔍 Swarm Position Explorer</div>',
                        unsafe_allow_html=True)
            iter_sel = st.slider(
                "Iteration", 0, len(r0.all_positions) - 1,
                len(r0.all_positions) - 1, key="iter_slider"
            )
            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.plotly_chart(
                    plot_swarm_snapshot(r0, iter_sel),
                    use_container_width=True,
                )
            with col_b:
                st.markdown('<div class="section-title">📋 Iteration Summary</div>',
                            unsafe_allow_html=True)
                pos = r0.all_positions[iter_sel]
                fit_at = [rosenbrock_constrained(pos[i]) for i in range(len(pos))]
                st.markdown(f"""
| Metric | Value |
|---|---|
| Iteration | **{iter_sel + 1} / {r0.n_iterations}** |
| Best Fitness at iter | **{r0.convergence[iter_sel]:.8f}** |
| # Particles | **{len(pos)}** |
| x₁ range | [{pos[:,0].min():.3f}, {pos[:,0].max():.3f}] |
| x₂ range | [{pos[:,1].min():.3f}, {pos[:,1].max():.3f}] |
| Mean fitness at iter | {np.mean(fit_at):.4f} |
""")

# ── Task 2 tab ──
with tab_task2:
    cmp = st.session_state.get("cmp")
    results2 = st.session_state.get("results")

    if cmp is None and results2 is not None and len(results2) == 2:
        # Both algorithms were run but comparison wasn't stored
        pass

    if cmp is None:
        st.info("👈 Select **Both (Compare)** in the sidebar and click **Run PSO** to see the comparison.")
    else:
        g_s = cmp["gbest_stats"]
        l_s = cmp["lbest_stats"]

        st.markdown('<div class="section-title">Side-by-side Statistics</div>',
                    unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                '<span class="badge badge-gbest">⭐ gbest PSO (Star Topology)</span>',
                unsafe_allow_html=True)
            st.markdown(f"""
| Metric | Value |
|---|---|
| Mean best fitness | `{g_s['best_fitness_mean']:.8f}` |
| Std deviation | `{g_s['best_fitness_std']:.8f}` |
| Best (min) fitness | `{g_s['best_fitness_min']:.8f}` |
| Mean runtime | `{g_s['time_mean']*1000:.2f} ms` |
| Evaluations / run | `{g_s['evaluations']:,}` |
""")
        with col2:
            st.markdown(
                '<span class="badge badge-lbest">🔗 lbest PSO (Ring Topology)</span>',
                unsafe_allow_html=True)
            st.markdown(f"""
| Metric | Value |
|---|---|
| Mean best fitness | `{l_s['best_fitness_mean']:.8f}` |
| Std deviation | `{l_s['best_fitness_std']:.8f}` |
| Best (min) fitness | `{l_s['best_fitness_min']:.8f}` |
| Mean runtime | `{l_s['time_mean']*1000:.2f} ms` |
| Evaluations / run | `{l_s['evaluations']:,}` |
""")

        # Convergence comparison (single run)
        if results2 and len(results2) == 2:
            st.plotly_chart(plot_convergence(results2), use_container_width=True, key="convergence_task2")

        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(plot_comparison_bar(cmp), use_container_width=True)
        with col4:
            st.plotly_chart(plot_multi_run_box(cmp), use_container_width=True)

        # Written analysis
        winner_fit  = "gbest" if g_s["best_fitness_mean"] < l_s["best_fitness_mean"] else "lbest"
        winner_time = "gbest" if g_s["time_mean"] < l_s["time_mean"] else "lbest"

        st.markdown('<div class="section-title">📝 Observations & Analysis</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
**Convergence Speed:**
gbest PSO propagates the global best to all particles simultaneously (star topology),
leading to faster initial convergence. lbest PSO uses a ring topology where information
spreads only to adjacent neighbours, causing slower but more exploratory convergence.

**Solution Quality:**
Based on {n_runs} independent runs, **{winner_fit} PSO** achieved a lower mean best fitness,
suggesting better solution quality on this run configuration.

**Runtime Complexity:**
Both algorithms share the same *O(n × d × T)* time complexity, where *n* is the swarm size,
*d* is the number of dimensions, and *T* is the number of iterations. The lbest variant
has a slightly higher constant factor due to neighbourhood lookup.
**{winner_time} PSO** was faster in practice.

**Premature Convergence:**
gbest PSO is more susceptible to premature convergence because all particles
are immediately attracted to the same global best. lbest PSO maintains higher
diversity through localised information sharing.

**Parameter Sensitivity:**
Both algorithms are sensitive to the inertia weight *w* (set to {w:.2f}) and acceleration
coefficients c₁={c1:.1f}, c₂={c2:.1f}. The sum c₁+c₂ should ideally be < 4 to ensure stability.
""")