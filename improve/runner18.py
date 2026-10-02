import subprocess, sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); tgt, fl = sys.argv[1], sys.argv[2]
with open(f"{HERE}/runner18_{tgt}_log.txt", "a") as log:
    log.write(f"=== {tgt} <- {tgt}_hist variant {fl}\n"); log.flush()
    subprocess.run([sys.executable, f"{HERE}/run_x.py", tgt, f"{tgt}_hist", "0,1,2", fl], stdout=log, stderr=subprocess.STDOUT)
open(f"{HERE}/runner18_{tgt}_done.txt", "w").write("done")
