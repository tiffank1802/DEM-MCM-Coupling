from directory import(
    BUCKET_BASE,
    BUCKET_ID,
    BUCKET_PREFIX,
)
from huggingface_hub import HfFileSystem
fs=HfFileSystem()


dossiers=fs.ls(f"hf://buckets/{BUCKET_ID}/{BUCKET_PREFIX}")
path=""
for dossier in dossiers:
    if all(j in dossier.get('name', '') for j in ["voronoi_", "dt1", "40cells"]):
        path=dossier['name']
import numpy as np
import matplotlib.pyplot as plt
P=np.array([])
with fs.open(f'{path}/transitionmatrix.npy','rb') as f:
    P=np.load(f)
    print(P)

fig,ax=plt.subplots()
fig.set_size_inches(10,10)
ax.imshow(P)
for i in range(P.shape[0]):
    for j in range(P.shape[0]):
        ax.text(j,i,np.round(P[i,j],3),ha='center',va='center',fontsize=5)

ax.set_title(f'{path.replace("buckets/ktongue/DEM_MCM/Experiments/","")}')
fig.tight_layout()
plt.savefig("transition.png")

# with fs.open(f"{BUCKET_BASE}/{BUCKET_PREFIX}/.keep" ,"w") as f:
#     f.write("")
   
