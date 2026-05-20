import os
import shutil
from collections import defaultdict


# Path to your cloned tika repository
TIKA_REPO_PATH = "/path/to/tika/repository"

# RSF file path
RSF_FILE_PATH = "/path/to/clusters.rsf"

# Output directory
OUTPUT_DIR = "./cluster_output"


# ==============================
# PARSE RSF FILE
# ==============================

def parse_rsf(rsf_file):
    """
    Parses RSF file and returns:
    {
        cluster_name: [fully.qualified.ClassName]
    }
    """

    clusters = defaultdict(list)

    with open(rsf_file, "r") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            # Expected format:
            # contain <cluster> <class>
            if len(parts) != 3:
                print(f"Skipping invalid line: {line}")
                continue

            relation, cluster_name, class_name = parts

            if relation != "contain":
                continue

            clusters[cluster_name].append(class_name)

    return clusters


# ==============================
# BUILD JAVA FILE INDEX
# ==============================

def build_java_file_index(repo_path):
    """
    Creates mapping:
    fully.qualified.ClassName -> absolute java file path
    """

    java_index = {}

    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".java"):

                full_path = os.path.join(root, file)

                relative_path = os.path.relpath(full_path, repo_path)

                # Remove .java extension
                relative_path = relative_path[:-5]

                # Convert path separator to dots
                class_name = relative_path.replace(os.sep, ".")

                # Remove src.main.java or src.test.java if present
                for prefix in [
                    "src.main.java.",
                    "src.test.java."
                ]:
                    if prefix in class_name:
                        class_name = class_name.split(prefix, 1)[1]

                java_index[class_name] = full_path

    return java_index


# ==============================
# CREATE CLUSTER DIRECTORIES
# ==============================

def generate_cluster_folders(clusters, java_index, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    missing_classes = []

    for cluster_name, class_list in clusters.items():

        cluster_dir = os.path.join(output_dir, cluster_name)

        os.makedirs(cluster_dir, exist_ok=True)

        print(f"\nProcessing cluster: {cluster_name}")

        for class_name in class_list:

            if class_name not in java_index:
                print(f"  Missing: {class_name}")
                missing_classes.append(class_name)
                continue

            source_file = java_index[class_name]

            destination_file = os.path.join(
                cluster_dir,
                os.path.basename(source_file)
            )

            shutil.copy2(source_file, destination_file)

            print(f"  Copied: {class_name}")

    return missing_classes


# ==============================
# MAIN
# ==============================

def main():

    print("Parsing RSF file...")
    clusters = parse_rsf(RSF_FILE_PATH)

    print("Building Java file index...")
    java_index = build_java_file_index(TIKA_REPO_PATH)

    print("Generating cluster folders...")
    missing = generate_cluster_folders(
        clusters,
        java_index,
        OUTPUT_DIR
    )

    print("\n==============================")
    print("Done")
    print("==============================")

    if missing:
        print("\nMissing Classes:")
        for cls in missing:
            print(f" - {cls}")


if __name__ == "__main__":
    main()