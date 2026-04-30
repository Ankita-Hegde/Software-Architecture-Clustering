from collections import defaultdict

selected_clusters = [
    "org.apache.tika.parser.ss",
    "org.apache.tika.sax.ss",
    "org.apache.tika.metadata.ss",
    "org.apache.tika.mime.ss",
    "org.apache.tika.detect.ss",
]

input_file = "clustering-output/tika_core_ACDC.rsf"
output_file = "rq2_cluster_samples.txt"

clusters = defaultdict(list)

with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 3:
            cluster = parts[1]
            item = parts[2]
            clusters[cluster].append(item)

with open(output_file, "w", encoding="utf-8") as out:
    for cluster in selected_clusters:
        out.write("=" * 70 + "\n")
        out.write(f"Cluster: {cluster}\n")
        out.write(f"Total classes: {len(clusters[cluster])}\n")
        out.write("Sample classes:\n")
        for item in clusters[cluster][:15]:
            out.write(f"- {item}\n")
        out.write("\n")

print(f"Created {output_file}")