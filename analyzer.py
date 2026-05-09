# This tool runs bash commands to compare A2A and CVG tests via Arcade Tools to run similarity checks in RSF files.

import subprocess
import pathlib

proj_root = str(pathlib.Path(__file__).resolve().parent)
a2a_path = proj_root + "/Arcade_Tools_W2/arcade_core_A2a.jar"
cvg_path = proj_root + "/Arcade_Tools_W2/arcade_core_Cvg.jar"

wca_output = proj_root + "/output/tika-b6bcce634_UEM_50_clusters.rsf"
limbo_output = proj_root + "/output/tika-b6bcce634_IL_50_clusters.rsf"
acdc_output = proj_root + "/output/tika-acdc.rsf"

def pair(algo_path, output1, output2):
    result = subprocess.run(f"java -jar {algo_path} {output1} {output2}", shell=True, capture_output=True, text=True)
    return result.stdout.strip()

#WCA and LIMBO
print(f"WCA vs LIMBO: a2a={pair(a2a_path,wca_output,limbo_output)}, cvg={pair(cvg_path,wca_output,limbo_output)}")
print(f"WCA vs ACDC: a2a={pair(a2a_path,wca_output,acdc_output)}, cvg={pair(cvg_path,wca_output,acdc_output)}")
print(f"LIMBO vs ACDC: a2a={pair(a2a_path,limbo_output,acdc_output)}, cvg={pair(cvg_path,limbo_output,acdc_output)}")