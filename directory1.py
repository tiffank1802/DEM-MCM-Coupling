from directory import(
    BUCKET_BASE,
    BUCKET_ID,
    BUCKET_PREFIX,
)
from huggingface_hub import HfFileSystem
fs=HfFileSystem()

subdirs=[
    "ResultsDtMCM",
"NewResultsMCM",
"RaffinageTemporel",
]
for folder in subdirs:
    
    fs.rm(f"{BUCKET_BASE}/{BUCKET_PREFIX}/{folder}",recursive=True)
    # fs.move(f"{BUCKET_BASE}/{folder}/*",f"{BUCKET_BASE}/{BUCKET_PREFIX}/{folder}")
   
