from huggingface_hub import HfApi,HfFileSystem 

fs=HfFileSystem()

BUCKET_ID = "ktongue/DEM_MCM"

BUCKET_PREFIX = "Experiments"
# subdirs=[
#     "ResultsDtMCM",
# "NewResultsMCM",
# "RaffinageTemporel",
# ]

BUCKET_BASE = f"hf://buckets/{BUCKET_ID}"

# for folder in subdirs:
    
#     fs.mkdirs(f"{BUCKET_BASE}/{BUCKET_PREFIX}/{folder}")
#     fs.move(f"{BUCKET_BASE}/{folder}/*",f"{BUCKET_BASE}/{BUCKET_PREFIX}/{folder}")
    


# # fs.makedirs(BUCKET_BASE,exist_ok=True)

# # with fs.open(f"{BUCKET_BASE}/.keep" ,"w") as f:
# #     f.write("")

if __name__=="__main__":
    

    from DEM_MCM.src.analyze_results import MarkovAnalyzer
    from DEM_MCM.src.partitioners import create_partitioner  # Utilisez create_partitioner !
    from DEM_MCM.src import run_sweep as r_s
    import numpy as np
    rsd_history={}
    Time=120
    folder_name = "adaptive_y_cylindrical_top1_bot36_split0.7_modequantile_NLT10_step10_dt2_tau50_start250"
    # 1. Créer analyzer et charger DEM
    analyzer = MarkovAnalyzer()
    analyzer.load_single_folder(folder_name) # charge la matrice de transition
    for Time in range(10,11):
        
        analyzer.load_dem_snapshots(file_indices=[Time])
        analyzer.label_species()

    # 2. Récupérer les coordonnées pour le fit (tous les snapshots)
    # all_coords = np.vstack([snap["coords"] for snap in analyzer.dem_snapshots if snap["t"]==290])
        all_coords = np.vstack([snap["coords"] for snap in analyzer.dem_snapshots ])
    # print(f"Shape des coordonnées pour fit: {all_coords.shape}")

    # 3. Créer ET fitter le partitionneur (3 cellules = nr=3, ntheta=1, nz=1)
  
    
        part = create_partitioner(method="adaptive",y_split=0.9,bottom_method="cartesian",bottom_kwargs= {
      "nx": 3,
      "ny": 5,
      "nz": 1
    })
        part.fit(all_coords)  # ← CRITIQUE : fit avec coordonnées réelles
    # print(f"Partitionneur fitté: {part.n_cells} cellules, label={part.label}")

    # 4. Vérifier que les centres sont calculés
    # print(f"x_center={getattr(part, '_x_center', 'MISSING')}")
    # print(f"y_center={getattr(part, '_y_center', 'MISSING')}")

    # 5. Calculer RSD
        rsd = analyzer.compute_rsd(folder_name=folder_name, partitioner=part,initial_time=Time)
        # rsd_history[f"init={Time}"]=analyzer.rsd
    
        print(f"RSD initial: {rsd['rsd_initial']:.3f}")
        print(f"RSD final: {rsd['rsd_final']:.3f}")
        print(f"t_50%: {rsd['mixing_time_50']}")
        print(f"t_90%: {rsd['mixing_time_90']}")
        print(analyzer.concentration_history.sum(1))
        print(analyzer.concentration_history.sum(1).shape)