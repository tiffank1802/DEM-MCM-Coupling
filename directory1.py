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

    # fs.move(f"{BUCKET_BASE}/{folder}/*",f"{BUCKET_BASE}/{BUCKET_PREFIX}/{folder}")
fs.rm(f"{BUCKET_BASE}/{BUCKET_PREFIX}ada*",recursive=True)
# with fs.open(f"{BUCKET_BASE}/{BUCKET_PREFIX}/.keep" ,"w") as f:
#     f.write("")
   
