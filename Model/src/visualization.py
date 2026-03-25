from __future__ import annotations

import matplotlib.pyplot as plt


def print_od_matrix(od_counts: dict) -> None:
    print("\n" + "=" * 60)
    print("FULLSTENDIG ORIGIN-DESTINATION (O-D) MATRIX")
    print("=" * 60)

    if not od_counts:
        print("Ingen O-D-hendelser registrert.")
        return

    for dest in sorted(od_counts.keys()):
        print(f"\n{dest} (destinasjon):")
        for origin in sorted(od_counts[dest].keys()):
            cls_counts = od_counts[dest][origin]
            parts = ", ".join(f"{n} {cls}" for cls, n in sorted(cls_counts.items()))
            print(f"  fra {origin}: {parts}")


def plot_od_routes(od_counts: dict) -> None:
    route_totals = {}
    route_by_class = {}
    all_classes = set()

    for dest, origins in od_counts.items():
        for origin, cls_counts in origins.items():
            route = f"{origin}→{dest}"
            total = int(sum(int(v) for v in cls_counts.values()))
            route_totals[route] = route_totals.get(route, 0) + total
            route_by_class.setdefault(route, {})

            for cls_name, n in cls_counts.items():
                n = int(n)
                route_by_class[route][cls_name] = route_by_class[route].get(cls_name, 0) + n
                if n > 0:
                    all_classes.add(cls_name)

    if not route_totals:
        print("Ingen O-D hendelser å plotte.")
        return

    routes = sorted(route_totals.keys())
    counts = [route_totals[r] for r in routes]

    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(routes, counts)
    plt.title("Antall kjøretøy per kjøreretning (O-D)")
    plt.xlabel("Kjøreretning")
    plt.ylabel("Antall")
    plt.grid(axis="y", alpha=0.3)

    ymax = max(counts) if counts else 1
    plt.ylim(0, ymax * 1.15 + 0.5)
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ymax * 0.02, str(count), ha="center", va="bottom")

    plt.tight_layout()
    plt.show()

    classes = sorted(all_classes)
    bottoms = [0] * len(routes)

    plt.figure(figsize=(10, 5))
    for cls_name in classes:
        values = [route_by_class.get(route, {}).get(cls_name, 0) for route in routes]
        plt.bar(routes, values, bottom=bottoms, label=cls_name)
        bottoms = [b + v for b, v in zip(bottoms, values)]

    plt.title("Kjøreretning med klassefordeling")
    plt.xlabel("Kjøreretning")
    plt.ylabel("Antall")
    plt.grid(axis="y", alpha=0.3)

    ymax2 = max(bottoms) if bottoms else 1
    plt.ylim(0, ymax2 * 1.15 + 0.5)
    for i, total in enumerate(bottoms):
        plt.text(i, total + ymax2 * 0.02, str(total), ha="center", va="bottom")

    plt.legend(title="Klasse", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()
