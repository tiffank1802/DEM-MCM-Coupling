from huggingface_hub import HfFileSystem, HfApi

fs  = HfFileSystem()
api = HfApi()

BUCKET_ID = "ktongue/DEM_MCM"
BASE_PATH  = "_Good/Experiment"
SKIP       = {"postraitement"}

base_hf = f"buckets/{BUCKET_ID}/{BASE_PATH}"
items   = [i for i in fs.ls(base_hf, detail=True) if i["type"] == "directory"]
print(f"📦 {len(items)} dossiers détectés\n")

ok = 0
for item in items:
    name = item["name"].split("/")[-1]

    if name in SKIP:
        print(f"⏭️  Skip '{name}'")
        continue

    print(f"🗑️  Suppression de '{name}'...", end="  ", flush=True)
    try:
        # Lister tous les fichiers du dossier
        files = [
            f.split(f"buckets/{BUCKET_ID}/", 1)[1]
            for f in fs.glob(f"{base_hf}/{name}/**", detail=False)
            if not fs.isdir(f)
        ]
        if not files:
            print("⚠️  vide")
            continue
        api.batch_bucket_files(bucket_id=BUCKET_ID, delete=files)
        print(f"✅  ({len(files)} fichiers supprimés)")
        ok += 1
    except Exception as e:
        print(f"❌ {e}")

print(f"\n✅ {ok} dossiers supprimés.")