"""Evaluation utilities for comparing predicted and ground-truth O-D results."""

from __future__ import annotations
import numpy as np
import pandas as pd

from src.GT import CLASS_ORDER, FLOW_ORDER

def map_class_name(name: str) -> str | None:
    """Map model-specific class labels to the thesis class taxonomy."""

    n = str(name).strip().lower()

    aliases = {
        "car": {"car"},
        "light truck": {"light truck"},
        "heavy truck": {"heavy truck", "truck"},
        "semi-trailer / combination vehicle": {
            "semi-trailer / combination vehicle",
            "semi trailer / combination vehicle",
            "semi-trailer combination vehicle",
            "semi trailer combination vehicle",},
        "bus": {"bus"},
        "motorcycle": {"motorcycle", "motorbike"},
        "bicycle": {"bicycle", "bike", "cycle", "cyclist", "bicyclist"},
    }

    for target, vals in aliases.items():
        if n in vals:
            return target
    return None


def build_pred_table_from_results(
    results_obj: dict,
    flow_order: list[str] = FLOW_ORDER,
    class_order: list[str] = CLASS_ORDER,
) -> pd.DataFrame:
    """Convert a results dictionary into a route-by-class prediction table."""

    pred = pd.DataFrame(0, index=flow_order, columns=class_order, dtype=int)
    od = results_obj.get("od_counts", {})

    for dest, origins in od.items():
        for origin, cls_counts in origins.items():
            route = f"{origin}->{dest}"
            if route not in pred.index:
                continue

            for cls_name, count in cls_counts.items():
                col = map_class_name(cls_name)
                if col is not None and col in pred.columns:
                    pred.loc[route, col] += int(count)

    pred["Sum"] = pred[class_order].sum(axis=1)
    return pred

def display_model_results(
    gt_df: pd.DataFrame,
    all_pred_tables: dict,
    all_diff_tables: dict,
    summary_df: pd.DataFrame,
    flow_order: list = None,
    class_order: list = None,
) -> None:
    """Display GT, predictions, differences, and metrics per model/tracker combination."""
    from IPython.display import display

    sorted_keys = sorted(all_pred_tables.keys(), key=lambda k: (k[0], k[1]))

    print("\n=== GROUND TRUTH ===")
    display(gt_df)

    for key in sorted_keys:
        pred_df = all_pred_tables[key]
        model_name, tracker_name = key
        diff_df = all_diff_tables[key]

        print(f"\n{'='*60}")
        print(f"  MODEL: {model_name}  |  TRACKER: {tracker_name}")
        print(f"{'='*60}")

        print("\n--- PREDICTION TABLE ---")
        display(pred_df)

        print("\n--- DIFF TABLE (pred - gt) ---")
        display(diff_df)

        # Additionally display class-error (%) aggregated across all routes
        try:
            gt_totals = gt_df[class_order].sum()
            diff_totals = abs(diff_df[class_order]).sum()
            class_err_pct = (diff_totals / gt_totals.replace(0, np.nan) * 100.0).round(1)
            class_err_pct = pd.DataFrame([class_err_pct], index=["All routes"])
            print("\n--- CLASS ERROR (ALL ROUTES, % of GT) ---")
            display(class_err_pct)
        except Exception:
            pass

        row = summary_df[
            (summary_df["model"] == model_name) &
            (summary_df["tracker"] == tracker_name)
        ]
        if not row.empty:
            rr = row.iloc[0]
            
            # Recalculate perfect_routes from diff_df (strict matching: ALL classes must be 0)
            if flow_order is not None and class_order is not None:
                perfect = ((diff_df.loc[flow_order, class_order] == 0).all(axis=1)).sum()
            else:
                # Fallback: use all rows
                perfect = ((diff_df[diff_df.columns[:-1]] == 0).all(axis=1)).sum()
            
            print(
                f"\n  sum_mae={rr['sum_mae']:.2f}  |  cell_mae={rr['cell_mae']:.2f}  |  "
                f"class_rel_err={rr.get('class_relative_error', float('nan')):.1f}%  |  "
                f"route_rel_err={rr.get('route_relative_error', float('nan')):.1f}%  |  "
                f"total_pred={int(rr['total_pred'])}  |  total_gt={int(rr['total_gt'])}  |  "
                f"total_err={int(rr['total_err'])}  |  total_abs_pct_err={rr['total_abs_pct_err']:.1f}%  |  "
                f"perfect_routes={int(perfect)}"
            )


def plot_class_distribution_detailed(
    all_pred_tables: dict,
    all_diff_tables: dict,
    gt_df: pd.DataFrame,
    flow_order: list[str] = FLOW_ORDER,
    class_order: list[str] = CLASS_ORDER,
    video_type: str = "DAY",
    all_results: dict | None = None,
    gt_persons: int | None = None,
    person_label: str = "Persons",
    person_zone_name: str = "Gangfeltet",
) -> None:
    """Plot detailed class composition and distribution analysis with 4 subplots.
    
    Creates:
    1. Total detections by class (GT vs all predictions)
    2. Per-class percentage error
    3. Route-level accuracy (perfect routes)
    4. Summary metrics table
    """
    import matplotlib.pyplot as plt
    
    print("\n" + "="*80)
    print(f"{video_type} - CLASS COMPOSITION & DISTRIBUTION")
    print("="*80)
    
    include_persons = all_results is not None and gt_persons is not None
    plot_labels = list(class_order) + ([person_label] if include_persons else [])
    display_labels = [
        "Semi-trailer" if label.lower().startswith("semi-trailer") else label
        for label in plot_labels
    ]

    def _person_total_for_key(key: tuple[str, str]) -> int:
        if not include_persons:
            return 0
        results_obj = all_results.get(key, {}) if isinstance(all_results, dict) else {}
        person_counts = results_obj.get("person_counts", {}) if isinstance(results_obj, dict) else {}
        if isinstance(person_counts, dict):
            return int(sum(int(v) for v in person_counts.values()))
        return 0

    def _cell_mae_for_key(key: tuple[str, str]) -> float:
        diff_df = all_diff_tables[key]
        return float(abs(diff_df[class_order]).values.mean())

    def _sum_mae_for_key(key: tuple[str, str]) -> float:
        diff_df = all_diff_tables[key]
        if "Sum" in diff_df.columns:
            return float(abs(diff_df.loc[flow_order, "Sum"]).mean())
        return float(abs(diff_df.loc[flow_order, class_order].sum(axis=1)).mean())

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    
    # Plot 1: Overall class composition (GT vs all predictions stacked)
    ax = axes[0, 0]
    x = np.arange(len(plot_labels))
    n_series = len(all_pred_tables) + 1  # GT + each model/tracker
    width = 0.8 / max(n_series, 1)
    
    gt_totals = gt_df[class_order].sum()
    gt_plot_counts = gt_totals.values.astype(float).tolist()
    if include_persons:
        gt_plot_counts.append(float(gt_persons))
    all_plot_counts = [np.asarray(gt_plot_counts, dtype=float)]
    gt_offset = (0 - (n_series - 1) / 2) * width
    ax.bar(x + gt_offset, gt_plot_counts, width, label="GT", alpha=0.9, color="black", edgecolor="white", linewidth=1)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_pred_tables)))
    for i, ((model_name, tracker_name), pred_df) in enumerate(all_pred_tables.items()):
        pred_totals = pred_df[class_order].sum()
        pred_plot_counts = pred_totals.values.astype(float).tolist()
        if include_persons:
            pred_plot_counts.append(float(_person_total_for_key((model_name, tracker_name))))
        all_plot_counts.append(np.asarray(pred_plot_counts, dtype=float))
        series_idx = i + 1
        offset = (series_idx - (n_series - 1) / 2) * width
        ax.bar(x + offset, pred_plot_counts, width,
               label=f"{model_name.replace('.pt', '')} | {tracker_name}",
               alpha=1.0, color=colors[i])
    
    ax.set_ylabel("Total Count", fontweight="bold")
    ax.set_xlabel("Classes", fontweight="bold")
    ax.set_title(
        f"Total Detections by Class ({video_type})" + (" + Persons" if include_persons else ""),
        fontweight="bold",
        fontsize=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, rotation=45, ha="center")

    # If class counts are highly imbalanced (e.g., car >> others), use symlog for readability
    flat_counts = np.concatenate(all_plot_counts)
    nonzero = flat_counts[flat_counts > 0]
    if nonzero.size >= 2:
        imbalance_ratio = float(nonzero.max() / nonzero.min())
        if imbalance_ratio >= 20:
            ax.set_yscale("symlog", linthresh=1.0)
            ax.set_ylabel("Total Count (symlog)", fontweight="bold")
            # Removed symlog scale annotation text as requested

    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    
    # Plot 2: Per-class accuracy (% error per class)
    ax = axes[0, 1]
    model_configs = list(all_diff_tables.keys())
    bar_width = 0.8 / max(len(model_configs), 1)
    x = np.arange(len(plot_labels))
    
    # Use same color scheme as Plot 1
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_configs)))
    
    for i, (model_name, tracker_name) in enumerate(model_configs):
        diff_df = all_diff_tables[(model_name, tracker_name)]
        gt_counts = gt_df[class_order]
        
        # Calculate % error per class
        pct_errors = []
        for cls in class_order:
            gt_val = gt_counts[cls].sum()
            diff_val = abs(diff_df.loc[:, cls].sum())
            pct_error = (diff_val / gt_val * 100) if gt_val > 0 else 0
            pct_errors.append(pct_error)

        if include_persons:
            pred_persons = _person_total_for_key((model_name, tracker_name))
            person_diff = abs(pred_persons - int(gt_persons))
            person_pct_error = (person_diff / gt_persons * 100) if gt_persons > 0 else 0
            pct_errors.append(person_pct_error)
        
        offset = (i - (len(model_configs) - 1) / 2) * bar_width
        ax.bar(x + offset, pct_errors, bar_width,
               label=f"{model_name.replace('.pt', '')} | {tracker_name}",
               alpha=1.0, color=colors[i])
    
    ax.set_ylabel("% Error", fontweight="bold")
    ax.set_xlabel("Classes", fontweight="bold")
    ax.set_title(
        f"Per-Class Percentage Error ({video_type})" + (" + Persons" if include_persons else ""),
        fontweight="bold",
        fontsize=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, rotation=45, ha="center")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    
    # Plot 3: Route-level accuracy (perfect routes) - use best model by lowest sum MAE
    ax = axes[1, 0]
    
    # Find best model based on lowest sum MAE (tie-break on cell MAE)
    best_key = min(
        all_diff_tables.keys(),
        key=lambda k: (_sum_mae_for_key(k), _cell_mae_for_key(k)),
    )
    best_model_name, best_tracker_name = best_key
    
    # Get the color index for the best model to match Plot 1 and 2
    best_model_idx = list(all_diff_tables.keys()).index(best_key)
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_diff_tables)))
    best_model_color = colors[best_model_idx]
    
    route_accuracy = []
    for route in flow_order:
        perfect = (abs(all_diff_tables[best_key].loc[route, class_order]) == 0).all()
        route_accuracy.append(1 if perfect else 0)
    
    # Use solid colors: model color for perfect, dark gray for imperfect
    colors_routes = [best_model_color if acc == 1 else "#333333" for acc in route_accuracy]
    ax.bar(flow_order, route_accuracy, color=colors_routes, alpha=1.0, edgecolor="black", linewidth=2)
    ax.set_ylabel("Perfect Match (1=Yes, 0=No)", fontweight="bold")
    ax.set_xlabel("Route", fontweight="bold")
    ax.set_title(f"Perfect Route Matches ({video_type} - {best_model_name.replace('.pt', '')} | {best_tracker_name})", 
                 fontweight="bold", fontsize=12)
    ax.set_ylim([0, 1.2])
    ax.set_xticks(range(len(flow_order)))
    ax.set_xticklabels(flow_order, rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")
    
    # Add checkmarks and X marks on top of bars for clarity
    for i, (route, acc) in enumerate(zip(flow_order, route_accuracy)):
        ax.text(i, acc + 0.05, "✓" if acc == 1 else "✗", ha="center", fontsize=16, 
               color="green" if acc == 1 else "red", fontweight="bold")
    
    # Plot 4: Summary metrics table
    ax = axes[1, 1]
    ax.axis("off")
    
    summary_data = []
    for model_name, tracker_name in all_pred_tables.keys():
        pred_df = all_pred_tables[(model_name, tracker_name)]
        diff_df = all_diff_tables[(model_name, tracker_name)]
        
        total_gt = gt_df["Sum"].sum()
        total_pred = pred_df["Sum"].sum()
        pct_err = abs(total_pred - total_gt) / total_gt * 100 if total_gt > 0 else float("nan")

        # class relative error: total abs class error / total gt across classes
        total_abs_class_error = abs(diff_df[class_order]).values.sum()
        total_gt_classes = gt_df[class_order].values.sum()
        class_rel = (total_abs_class_error / total_gt_classes * 100.0) if total_gt_classes > 0 else float("nan")

        # route relative error: mean(|pred_route - gt_route| / gt_route) across routes with gt>0
        route_abs = abs(diff_df.loc[flow_order, class_order].sum(axis=1))
        route_gt = gt_df.loc[flow_order, "Sum"]
        valid_routes = route_gt > 0
        if valid_routes.any():
            per_route_rel = (route_abs[valid_routes] / route_gt[valid_routes]).astype(float)
            route_rel = float(per_route_rel.mean() * 100.0)
        else:
            route_rel = float("nan")

        person_pred = _person_total_for_key((model_name, tracker_name)) if include_persons else None
        person_err = (person_pred - int(gt_persons)) if include_persons else None
        person_pct_err = (abs(person_err) / gt_persons * 100) if include_persons and gt_persons > 0 else (np.nan if include_persons else None)

        summary_data.append(
            (
                class_rel,
                route_rel,
                pct_err,
                abs(person_err) if include_persons else 0,
                model_name.replace('.pt', ''),
                tracker_name,
                f"{class_rel:.1f}%" if not np.isnan(class_rel) else "-",
                f"{route_rel:.1f}%" if not np.isnan(route_rel) else "-",
                f"{pct_err:.1f}%" if not np.isnan(pct_err) else "-",
                f"{int(person_pred)}" if include_persons else None,
                f"{int(person_err):+d}" if include_persons else None,
                f"{person_pct_err:.1f}%" if include_persons and not np.isnan(person_pct_err) else "-" if include_persons else None,
            )
        )

    summary_data.sort(key=lambda row: (row[0], row[1], row[3], row[2], row[4], row[5]))

    max_tracker_len = max((len(str(row[5])) for row in summary_data), default=0)

    if include_persons:
        table_rows = [[row[4], row[5], row[6], row[7], row[8], row[10]] for row in summary_data]
        col_widths = [0.18, 0.18, 0.14, 0.14, 0.12, 0.12]
        if max_tracker_len >= 20:
            col_widths = [0.16, 0.28, 0.12, 0.12, 0.10, 0.10]
        table = ax.table(
            cellText=table_rows,
            colLabels=["Model", "Tracker", "Class Err", "Route Err", "Error %", "Person Err"],
            cellLoc="center",
            loc="center",
            colWidths=col_widths,
        )
    else:
        table_rows = [[row[4], row[5], row[6], row[7], row[8]] for row in summary_data]
        col_widths = [0.24, 0.24, 0.16, 0.16, 0.20]
        if max_tracker_len >= 20:
            col_widths = [0.18, 0.32, 0.14, 0.14, 0.22]
        table = ax.table(
            cellText=table_rows,
            colLabels=["Model", "Tracker", "Class Err", "Route Err", "Error %"],
            cellLoc="center",
            loc="center",
            colWidths=col_widths
        )
    table_font_size = 8 if max_tracker_len >= 20 else (9 if include_persons else 10)
    table.auto_set_font_size(False)
    table.set_fontsize(table_font_size)
    table.scale(1, 2)

    # Style header
    n_cols = 6 if include_persons else 5
    for i in range(n_cols):
        table[(0, i)].set_facecolor("#40466e")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Alternate row colors
    for i in range(1, len(table_rows) + 1):
        for j in range(n_cols):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#f0f0f0")

    ax.set_title(
        f"{video_type} Results Summary" + (" + Persons" if include_persons else ""),
        fontweight="bold",
        fontsize=12,
        pad=20,
    )
    
    plt.tight_layout()
    plt.show()
    
    print("\nClass composition analysis complete")


def run_experiments(
    models: list[str],
    trackers: dict[str, str | dict[str, str | None]],
    video_name: str,
    gt_df: pd.DataFrame,
    gt_persons: int = 0,
    session_label: str = "EXPERIMENT",
    video_type: str = "day",
    flow_order: list[str] = FLOW_ORDER,
    class_order: list[str] = CLASS_ORDER,
    zone_name: str = "Gangfeltet",
    setup_tracking_context=None,
    run_video_deploy=None,
    **run_video_deploy_kwargs,
) -> tuple[dict, dict, dict, pd.DataFrame, pd.DataFrame]:
    """
    Run tracking experiments for multiple models and trackers (vehicles + persons).
    
    Args:
        models: List of model names (e.g., ["YOLO26m_final.pt", "YOLO26n_final.pt"])
        trackers: Dict mapping tracker names to YAML paths
        video_name: Video file name
        gt_df: Ground truth dataframe for vehicles
        gt_persons: Ground truth person count (default: 0)
        session_label: Label for printing (e.g., "DAY", "NIGHT")
        video_type: Either "day" or "night" - determines which tracker parameter is used
        flow_order: Order of flow directions
        class_order: Order of classes
        zone_name: Zone to analyze for persons (default: "Gangfeltet")
        setup_tracking_context: Function to setup tracking context
        run_video_deploy: Function to run tracking
        **run_video_deploy_kwargs: Additional kwargs for run_video_deploy
    
    Returns:
        Tuple of (all_results, all_pred_tables, all_diff_tables, vehicles_summary_df, persons_summary_df)
    """
    print(f"Running {session_label} video experiments with run_video_deploy...\n")
    
    vehicle_rows = []
    persons_rows = []
    all_results = {}
    all_pred_tables = {}
    all_diff_tables = {}
    
    for model_name in models:
        for tracker_name, tracker_yaml in trackers.items():
            print(f"\n{'='*60}")
            print(f"{session_label} | {model_name} | {tracker_name}")
            print(f"{'='*60}")
            
            try:
                day_tracker_yaml = None
                night_tracker_yaml = None

                if isinstance(tracker_yaml, dict):
                    day_tracker_yaml = tracker_yaml.get("day")
                    night_tracker_yaml = tracker_yaml.get("night")
                elif video_type.lower() == "night":
                    night_tracker_yaml = tracker_yaml
                else:
                    day_tracker_yaml = tracker_yaml

                deploy_out = run_deploy(
                    model_name=model_name,
                    video_name=video_name,
                    gt_df=gt_df,
                    gt_persons=gt_persons,
                    tracker_name=tracker_name,
                    day_tracker_yaml=day_tracker_yaml,
                    night_tracker_yaml=night_tracker_yaml,
                    zone_name=zone_name,
                    flow_order=flow_order,
                    class_order=class_order,
                    setup_tracking_context=setup_tracking_context,
                    run_video_deploy=run_video_deploy,
                    **run_video_deploy_kwargs,
                )
                
                key = (model_name, tracker_name)
                all_results[key] = deploy_out["results"]
                all_pred_tables[key] = deploy_out["pred_df"]
                all_diff_tables[key] = deploy_out["diff_df"]
                vehicle_rows.append(deploy_out["vehicle_row"])
                persons_rows.append(deploy_out["persons_row"])
                
                print(f"Complete")
                
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
    
    # Build summary dataframes
    vehicles_summary_df = pd.DataFrame(vehicle_rows).sort_values(
        by=["sum_mae", "cell_mae", "total_abs_pct_err", "elapsed_s"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    
    persons_summary_df = pd.DataFrame(persons_rows)
    
    print("\n" + "="*80)
    print(f"{session_label} RESULTS SUMMARY - VEHICLES")
    print("="*80)
    
    return all_results, all_pred_tables, all_diff_tables, vehicles_summary_df, persons_summary_df


def run_deploy(
    model_name: str,
    video_name: str,
    gt_df: pd.DataFrame,
    gt_persons: int = 0,
    tracker_name: str = "tracker",
    day_tracker_yaml: str | None = None,
    night_tracker_yaml: str | None = None,
    zone_name: str = "Gangfeltet",
    flow_order: list[str] = FLOW_ORDER,
    class_order: list[str] = CLASS_ORDER,
    setup_tracking_context=None,
    run_video_deploy=None,
    **run_video_deploy_kwargs,
) -> dict:
    """
    Run one deploy experiment and return raw results + analysis dataframes/rows.

    Supports day tracker only, night tracker only, or both (dynamic switching).
    """
    from src.tracking_utils import setup_tracking_context as default_setup
    from src.run_video_deploy import run_video_deploy as default_deploy

    if day_tracker_yaml is None and night_tracker_yaml is None:
        raise ValueError("At least one of day_tracker_yaml or night_tracker_yaml must be provided")

    if setup_tracking_context is None:
        setup_tracking_context = default_setup
    if run_video_deploy is None:
        run_video_deploy = default_deploy

    cfg_primary = setup_tracking_context(
        model_name=model_name,
        tracker_yaml=day_tracker_yaml or night_tracker_yaml,
        video_name=video_name,
        apply_globals=False,
    )

    day_tracker = None
    night_tracker = None

    if day_tracker_yaml is not None:
        cfg_day = setup_tracking_context(
            model_name=model_name,
            tracker_yaml=day_tracker_yaml,
            video_name=video_name,
            apply_globals=False,
        )
        day_tracker = cfg_day["custom_tracker"]

    if night_tracker_yaml is not None:
        cfg_night = setup_tracking_context(
            model_name=model_name,
            tracker_yaml=night_tracker_yaml,
            video_name=video_name,
            apply_globals=False,
        )
        night_tracker = cfg_night["custom_tracker"]

    results = run_video_deploy(
        model=cfg_primary["model"],
        video_source=cfg_primary["video_source"],
        runs_dir=cfg_primary["runs_dir"],
        day_tracker=day_tracker,
        night_tracker=night_tracker,
        **run_video_deploy_kwargs,
    )

    pred_df = build_pred_table_from_results(results, flow_order, class_order)
    pred_df = pred_df.reindex(flow_order).fillna(0).astype(int)
    pred_df["Sum"] = pred_df[class_order].sum(axis=1)
    diff_df = pred_df[class_order + ["Sum"]] - gt_df[class_order + ["Sum"]]

    abs_diff = diff_df.abs()
    total_gt = int(gt_df["Sum"].sum())
    total_pred = int(pred_df["Sum"].sum())
    total_err = total_pred - total_gt
    total_abs_pct_err = (abs(total_err) / total_gt * 100.0) if total_gt > 0 else float("nan")

    # Class-level relative error: total absolute class-cell error / total GT across classes
    total_abs_class_error = abs_diff[class_order].values.sum()
    total_gt_classes = gt_df[class_order].values.sum()
    class_relative_error = (total_abs_class_error / total_gt_classes * 100.0) if total_gt_classes > 0 else float("nan")

    # Route-level relative error: mean(|pred_route - gt_route| / gt_route) across routes with gt>0
    route_abs = abs_diff[class_order].sum(axis=1)
    route_gt = gt_df["Sum"]
    valid_routes = route_gt > 0
    if valid_routes.any():
        per_route_rel = (route_abs[valid_routes] / route_gt[valid_routes]).astype(float)
        route_relative_error = float(per_route_rel.mean() * 100.0)
    else:
        route_relative_error = float("nan")

    vehicle_row = {
        "model": model_name,
        "tracker": tracker_name,
        "elapsed_s": results.get("elapsed_time", float("nan")),
        "unique_ids": results.get("unique_ids", float("nan")),
        "cell_mae": float(abs_diff[class_order].values.mean()),
        "sum_mae": float(abs_diff["Sum"].mean()),
        "class_relative_error": float(class_relative_error),
        "route_relative_error": float(route_relative_error),
        "total_gt": total_gt,
        "total_pred": total_pred,
        "total_err": total_err,
        "total_abs_pct_err": float(total_abs_pct_err),
        "perfect_routes": int(((abs_diff[class_order] == 0).all(axis=1)).sum()),
    }

    # Add Route Err % column to diff_df (right of 'Sum') - per-route relative error
    try:
        route_abs_series = abs_diff[class_order].sum(axis=1)
        route_gt_series = gt_df["Sum"]
        route_err_pct = pd.Series(index=diff_df.index, dtype=float)
        valid = route_gt_series > 0
        route_err_pct.loc[valid] = (route_abs_series[valid] / route_gt_series[valid]) * 100.0
        route_err_pct.loc[~valid] = float("nan")

        # Insert after Sum
        cols = list(diff_df.columns)
        insert_pos = cols.index("Sum") + 1 if "Sum" in cols else len(cols)
        diff_df.insert(insert_pos, "Route Err %", route_err_pct.round(1))
    except Exception:
        # If anything goes wrong, keep original diff_df
        pass

    zone_events = [
        od for od in results.get("od_events", [])
        if od.get("class", "").lower() == "person"
        and (od.get("origin") == zone_name or od.get("dest") == zone_name)
    ]
    predicted_persons = len(zone_events)
    person_error = predicted_persons - gt_persons
    person_pct_error = (abs(person_error) / max(gt_persons, 1) * 100.0) if gt_persons > 0 else float("nan")

    persons_row = {
        "model": model_name,
        "tracker": tracker_name,
        "gt_persons": gt_persons,
        "predicted_persons": predicted_persons,
        "error": person_error,
        "pct_error": person_pct_error,
    }

    return {
        "results": results,
        "pred_df": pred_df,
        "diff_df": diff_df,
        "vehicle_row": vehicle_row,
        "persons_row": persons_row,
    }


__all__ = [
    "map_class_name",
    "build_pred_table_from_results",
    "display_model_results",
    "plot_class_distribution_detailed",
    "run_deploy",
    "run_experiments",
]
