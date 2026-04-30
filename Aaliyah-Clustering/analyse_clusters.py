from collections import defaultdict

files = [
    "clustering-output-filtered/tika_filtered_ACDC.rsf",
    "clustering-output-filtered/tikafiltered-4.0_UEM_50_clusters.rsf",
    "clustering-output-filtered/tikafiltered-4.0_IL_50_clusters.rsf",
]

for file in files:
    clusters = defaultdict(list)

    with open(file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                cluster = parts[1]
                item = parts[2]
                clusters[cluster].append(item)

    sizes = sorted([len(v) for v in clusters.values()], reverse=True)

    print("=" * 50)
    print(file)
    print("Clusters:", len(clusters))
    print("Largest cluster:", max(sizes) if sizes else 0)
    print("Smallest cluster:", min(sizes) if sizes else 0)
    print("Top 10 sizes:", sizes[:10])