import subprocess
from pathlib import Path
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# --- CONFIGURATION ---
project_name = "boh-yai"
base_dir = Path("~/photogrammetry2").expanduser()
project_dir = base_dir / project_name
exports_dir = project_dir / "exports"
archive_dir = project_dir / "archive"
archive_dir.mkdir(exist_ok=True)

quality_levels = ["merged", "medium", "low"]

def archive_exists(label):
    base_archive = archive_dir / f"{label}.7z"
    split_archives = list(archive_dir.glob(f"{label}.7z.*"))
    return base_archive.exists() or split_archives

def compress(label, files):
    if archive_exists(label):
        log(f"⏭️ Skipping {label}: Archive already exists.")
        return

    files = [str(f) for f in files if f.exists()]
    if not files:
        log(f"⚠️ No source files found for: {label}")
        return

    log(f"📦 Compressing {label}...")
    cmd = ["7z", "a", "-mx=9", "-v5g", str(archive_dir / f"{label}.7z")] + files
    subprocess.run(cmd, check=True)
    log(f"✅ Compression complete: {label}")

# --- Compress per LOD quality ---
for quality in quality_levels:
    if quality == "merged":
        base = exports_dir / f"{project_name}-merged"
    else:
        base = exports_dir / f"{project_name}_{quality}"

    compress(f"{quality}_obj", [base.with_suffix(".obj"), base.with_suffix(".mtl")])
    compress(f"{quality}_glb", [base.with_suffix(".glb")])

# --- Compress entire project folder ---
if archive_exists(f"{project_name}_full_project"):
    log(f"⏭️ Skipping full project: Archive already exists.")
else:
    log("📦 Compressing full project directory...")
    subprocess.run([
        "7z", "a", "-mx=9", "-v5g", str(archive_dir / f"{project_name}_full_project.7z"),
        str(project_dir)
    ], check=True)
    log(f"✅ Full project compression complete.")
