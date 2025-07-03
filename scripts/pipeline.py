import Metashape
import argparse
import subprocess
from glob import glob
import os
import sys
import datetime
import time

# --- LOGGING ---
def log(msg):
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# --- PROGRESS CALLBACK ---
last_percent = -1

def progress_callback(p):
    global last_percent
    try:
        percent = round(p / 100, 1)
        if percent != last_percent:
            print(f"Progress: {percent}%")
            last_percent = percent
    except:
        pass

# --- EXTRACT FRAMES ---
def extract_frames(videos_dir, output_folder, fps=2):
    os.makedirs(output_folder, exist_ok=True)
    for filename in sorted(os.listdir(videos_dir)):
        if filename.lower().endswith(('.mp4', '.mov', '.360')):
            input_path = os.path.join(videos_dir, filename)
            base = os.path.splitext(filename)[0]
            output_path = os.path.join(output_folder, f"{base}_%04d.jpg")
            if not glob(output_path.replace('%04d', '*')):
                log(f"📽️ Extracting frames from {filename}...")
                subprocess.run([
                    "ffmpeg", "-i", input_path,
                    "-qscale:v", "2",
                    "-vf", f"fps={fps}",
                    output_path
                ], check=True)
            else:
                log(f"✅ Frames already extracted for {filename}, skipping...")

# --- INJECT 360 METADATA ---
def inject_360_metadata(output_folder, batch_size=200):
    log("🔁 Checking for existing 360° metadata...")
    all_files = sorted(glob(os.path.join(output_folder, "**/*.jpg"), recursive=True))

    if not all_files:
        log("⚠️ No JPG files found to tag.")
        return

    last_file = all_files[-1]
    try:
        meta_flag = subprocess.getoutput(f'exiftool -s3 -UsePanoramaViewer {last_file}').strip()
    except Exception as e:
        log(f"⚠️ Error reading metadata from {last_file}: {e}")
        meta_flag = ""

    if meta_flag == "True":
        log("✅ 360° metadata already present. Skipping tagging.")
        return

    log("🏷️  Adding 360° metadata to all .jpg files...")
    for i in range(0, len(all_files), batch_size):
        batch = all_files[i:i + batch_size]
        log(f"📦 Processing batch {i // batch_size + 1} of {(len(all_files) + batch_size - 1) // batch_size}")
        cmd = [
            "exiftool", "-overwrite_original",
            "-ProjectionType=equirectangular",
            "-UsePanoramaViewer=True"
        ] + batch
        subprocess.run(cmd, check=True)

# --- MAIN PHOTOGRAMMETRY PROCESS ---
def run_photogrammetry_pipeline(base_dir, project_name):
    project_dir = os.path.join(base_dir, project_name)
    frames_dir = os.path.join(project_dir, "frames")
    project_path = os.path.join(project_dir, f"{project_name}.psx")
    exports_dir = os.path.join(project_dir, "exports")
    os.makedirs(exports_dir, exist_ok=True)

    # --- CREATE AND ALIGN ---
    log("🧱 Creating Metashape project and aligning...")
    doc = Metashape.Document()
    if os.path.exists(project_path):
        doc.open(project_path)
        log("✅ Project loaded from disk.")
    else:
        doc.save(project_path)

    if not doc.chunks:
        chunk = doc.addChunk()
        chunk.label = "MainChunk"
        images = [os.path.join(frames_dir, f) for f in sorted(os.listdir(frames_dir)) if f.lower().endswith(".jpg")]
        chunk.addPhotos(images)
        chunk.sensors[0].type = Metashape.Sensor.Type.Spherical
        chunk.sensors[0].fixed = False

        chunk.matchPhotos(
            downscale=2,
            generic_preselection=True,
            reference_preselection=False,
            keypoint_limit=160000,
            tiepoint_limit=40000,
            progress=progress_callback
        )
        chunk.alignCameras(progress=progress_callback)
        doc.save()

    # --- SPLIT INTO CHUNKS PROPERLY (only if not already split) ---
    CHUNK_COUNT = 8

    if len(doc.chunks) >= CHUNK_COUNT:
        log("✅ Chunks already split. Skipping split step.")
    else:
        log(f"🔀 Splitting into {CHUNK_COUNT} chunks (copying tie points)...")
        chunk = doc.chunks[0]
        cameras = list(chunk.cameras)
        per_chunk = len(cameras) // CHUNK_COUNT

        for i in range(CHUNK_COUNT):
            new_chunk = chunk.copy()  # ✅ preserve alignment + tie points
            new_chunk.label = f"GPU-{i}"  # You can adjust label format if you want

            new_chunk.cameras.clear()

            start = i * per_chunk
            end = (i + 1) * per_chunk if i < CHUNK_COUNT - 1 else len(cameras)
            selected = set(cameras[start:end])
            for cam in chunk.cameras:
                if cam.label in {c.label for c in selected}:
                    new_chunk.cameras.append(cam)

            doc.chunks.append(new_chunk)

        doc.remove(chunk)
        doc.save()

    # --- BUILD POINT CLOUD ---
    for chunk in doc.chunks:
        log(f"🌫️  Checking point cloud for chunk: {chunk.label}")

        if not chunk.enabled or not chunk.cameras or all([not cam.transform for cam in chunk.cameras]):
            log(f"⚠️ Skipping {chunk.label} — no valid cameras or chunk disabled.")
            continue

        if chunk.depth_maps is None or not chunk.depth_maps:
            log(f"  ➤ Building depth maps for {chunk.label}...")
            chunk.buildDepthMaps(
                downscale=1,
                filter_mode=Metashape.MildFiltering,
                reuse_depth=True,
                progress=progress_callback
            )
            doc.save()

        #if not chunk.point_cloud:
            #log(f"  ➤ Building dense cloud for {chunk.label}...")
            #chunk.buildPointCloud(
                #point_colors=True,
                #keep_depth=True,
                #progress=progress_callback
            #)
            #doc.save()

    # --- BUILD MESH ---
    for chunk in doc.chunks:
        if not chunk.model:
            log(f"🕸️   Building mesh for chunk: {chunk.label}")
            chunk.buildModel(
                surface_type=Metashape.SurfaceType.Arbitrary,
                source_data=Metashape.DataSource.DepthMapsData,  # ✅ GPU-accelerated
                interpolation=Metashape.Interpolation.EnabledInterpolation,
                face_count=Metashape.FaceCount.LowFaceCount,
                vertex_colors=False,
                vertex_confidence=False,
                trimming_radius=0,
                progress=progress_callback
            )
            doc.save()

    # --- BUILD TEXTURE ---
    for chunk in doc.chunks:
        if not chunk.model:
            log(f"⚠️ Skipping texture: no model in chunk {chunk.label}")
            continue

        if len(chunk.model.textures) == 0:
            log(f"🧵 Building texture for chunk: {chunk.label}")

            log(f"  ➤ Building UV for {chunk.label}...")
            chunk.buildUV(
                mapping_mode=Metashape.MappingMode.GenericMapping,
                texture_size=8192,
                progress=progress_callback
            )

            log(f"  ➤ Building texture for {chunk.label}...")
            chunk.buildTexture(
                blending_mode=Metashape.BlendingMode.MosaicBlending,
                texture_size=8192,
                ghosting_filter=True,
                fill_holes=True,
                progress=progress_callback
            )

            doc.save()
        else:
            log(f"✅ Texture already exists for chunk: {chunk.label}, skipping.")


    # --- EXPORT ---
    log("📦 Preparing to merge and export chunks into single GLBs...")

    glb_dir = os.path.join(exports_dir, "glb_exports")
    os.makedirs(glb_dir, exist_ok=True)

    def export_merged_glb(decimate_ratio=None, label="full"):
        log(f"📦 Merging and exporting GLB for quality: {label}...")

        # Collect only chunks that have a valid model
        chunks_to_merge = []
        for chunk in doc.chunks:
            if not chunk.enabled or not chunk.model:
                log(f"⚠️ Skipping {chunk.label} — no model or chunk disabled.")
                continue
            if chunk.model and len(chunk.model.faces) > 0:
                chunks_to_merge.append(chunk)

        if not chunks_to_merge:
            log(f"❌ No chunks with valid models found for {label}. Skipping...")
            return

        log(f"🛠️ Merging {len(chunks_to_merge)} chunks...")
        doc.mergeChunks(chunks_to_merge, merge_markers=False)

        merged_chunk = doc.chunks[-1]  # ✅ get the newly created merged chunk manually

        if not merged_chunk or not merged_chunk.model:
            log(f"❌ Merge failed, no model produced for {label}.")
            return

        merged_chunk.label = f"Merged_{label}"
        if decimate_ratio is not None:
            log(f"🔻 Decimating merged model to {int(decimate_ratio * 100)}% faces...")
            target_faces = int(len(merged_chunk.model.faces) * decimate_ratio)
            merged_chunk.decimateModel(face_count=target_faces)

        output_path = os.path.join(glb_dir, f"{project_name}_{label}.glb")
        log(f"💾 Saving merged GLB to {output_path}...")
        merged_chunk.exportModel(
            path=output_path,
            format=Metashape.ModelFormatGLTF,
            binary=True,
            save_texture=True,
            save_uv=True,
            save_normals=True,
            texture_format=Metashape.ImageFormat.ImageFormatJPEG,
            draco_compression_level=6
        )

        log(f"🧹 Removing merged chunk after export...")
        doc.remove(merged_chunk)

        log(f"✅ Finished merged GLB export for quality: {label}.")

    # Export all three qualities
    export_merged_glb(decimate_ratio=None, label="full")
    export_merged_glb(decimate_ratio=0.15, label="medium")
    export_merged_glb(decimate_ratio=0.03, label="low")

    log("✅ All GLB exports complete.")

# --- ENTRY POINT ---
parser = argparse.ArgumentParser()
parser.add_argument("--project-name", required=True, help="Project name (used to create folders)")
parser.add_argument("--videos", required=True, help="Directory containing input 360 videos")
parser.add_argument("--fps", type=int, default=2, help="Frames per second to extract")
parser.add_argument("--output-dir", default="~/photogrammetry", help="Base output directory")
args = parser.parse_args()

args.output_dir = os.path.expanduser(args.output_dir)
project_dir = os.path.join(args.output_dir, args.project_name)
frames_dir = os.path.join(project_dir, "frames")

# --- RUN ---
extract_frames(args.videos, frames_dir, fps=args.fps)
inject_360_metadata(frames_dir)
run_photogrammetry_pipeline(args.output_dir, args.project_name)
