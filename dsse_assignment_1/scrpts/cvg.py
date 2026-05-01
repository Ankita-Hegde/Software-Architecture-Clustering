import sys
from collections import defaultdict

def read_rsf_dependencies(file_path):
    deps = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                _, src, dst = parts[:3]
                deps.append((src, dst))
    return deps


def read_clusters(file_path):
    clusters = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                _, cluster, entity = parts[:3]
                clusters[entity] = cluster
    return clusters


def compute_cvg(deps, clusters):
    internal = 0
    total = len(deps)

    for src, dst in deps:
        if src in clusters and dst in clusters:
            if clusters[src] == clusters[dst]:
                internal += 1

    return internal / total if total > 0 else 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python cvg.py <dependency.rsf> <clustering.rsf>")
        sys.exit(1)

    dep_file = sys.argv[1]
    cluster_file = sys.argv[2]

    deps = read_rsf_dependencies(dep_file)
    clusters = read_clusters(cluster_file)

    cvg_score = compute_cvg(deps, clusters)
    print(f"CVG: {cvg_score}")