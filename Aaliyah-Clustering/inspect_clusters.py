from collections import defaultdict

files = {
    "ACDC_FILTERED": "clustering-output-filtered/tika_filtered_ACDC.rsf",
    "WCA_FILTERED": "clustering-output-filtered/tikafiltered-4.0_UEM_50_clusters.rsf",
    "LIMBO_FILTERED": "clustering-output-filtered/tikafiltered-4.0_IL_50_clusters.rsf",
}

for algo, file in files.items():
    clusters = defaultdict(list)

    with open(file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                cluster = parts[1]
                item = parts[2]
                clusters[cluster].append(item)

    print("\n" + "=" * 80)
    print(algo, "-", file)
    print("=" * 80)

    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)

    for cluster, items in sorted_clusters[:10]:
        print(f"\nCluster: {cluster}")
        print(f"Size: {len(items)}")
        print("Sample classes:")
        for item in items[:15]:
            print("  -", item)