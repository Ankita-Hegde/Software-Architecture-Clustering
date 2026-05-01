from collections import defaultdict
import sys

def load_rsf(path):
    clusters = defaultdict(set)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            _, cluster, element = parts
            clusters[cluster].add(element)
    return clusters

def jaccard(set1, set2):
    return len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0

def a2a(file1, file2):
    c1 = load_rsf(file1)
    c2 = load_rsf(file2)

    scores = []

    for k1 in c1:
        best = 0
        for k2 in c2:
            best = max(best, jaccard(c1[k1], c2[k2]))
        scores.append(best)

    return sum(scores) / len(scores) if scores else 0

if __name__ == "__main__":
    f1, f2 = sys.argv[1], sys.argv[2]
    print("A2A similarity:", a2a(f1, f2))