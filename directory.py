from huggingface_hub import HfApi,HfFileSystem 
import matplotlib.pyplot as plt

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
    Time=250
    # folder_name ="physics_400cells_pos_NLT10_step10_dt2_tau50_start250"
    # folder_name ="cartesian_nx3_ny3_nz3_NLT10_step100_dt2_tau50_start250"
    # folder_name ="voronoi_600cells_NLT10_step10_dt2_tau50_start250"
    folder_name ="voronoi_1000cells_NLT10_step10_dt2_tau50_start250"
    # folder_name ="cylindrical_nr3_nth8_nz1_equal_area_NLT10_step10_dt2_tau100_start250"
    # folder_name ="voronoi_400cells_NLT10_step10_dt2_tau50_start250"
    # 1. Créer analyzer et charger DEM
    analyzer = MarkovAnalyzer()
    analyzer.load_single_folder(folder_name) # charge la matrice de transition
    for Time in range(250,251):
        
        analyzer.load_dem_snapshots(file_indices=[Time])
        analyzer.label_species()

    # 2. Récupérer les coordonnées pour le fit (tous les snapshots)
    # all_coords = np.vstack([snap["coords"] for snap in analyzer.dem_snapshots if snap["t"]==290])
        all_coords = np.vstack([snap["coords"] for snap in analyzer.dem_snapshots ])
    # print(f"Shape des coordonnées pour fit: {all_coords.shape}")

    # 3. Créer ET fitter le partitionneur (3 cellules = nr=3, ntheta=1, nz=1)
  
    
    #     part = create_partitioner(method="adaptive",y_split=0.9,bottom_method="cartesian",bottom_kwargs= {
    #   "nx": 3,
    #   "ny": 5,
    #   "nz": 1
    # })
        # part = create_partitioner(method="physics",n_cells=400
    # )
    #     part = create_partitioner(method="cartesian",nx=3,ny=3,nz=3
    # )
    #     part = create_partitioner(method="cylindrical",nr=3,ntheta=8,nz=1
    # )
        part = create_partitioner(method="voronoi",n_cells=1000
    )
        part.fit(all_coords)  # ← CRITIQUE : fit avec coordonnées réelles
    # print(f"Partitionneur fitté: {part.n_cells} cellules, label={part.label}")

    # 4. Vérifier que les centres sont calculés
    # print(f"x_center={getattr(part, '_x_center', 'MISSING')}")
    # print(f"y_center={getattr(part, '_y_center', 'MISSING')}")

    # 5. Calculer RSD
        step=10
        tau=50
        tt=step+tau
        analyzer.label_species()
        rsd = analyzer.compute_rsd(folder_name=folder_name, partitioner=part,initial_time=Time,n_steps=39)
        rsd_history[f"init={Time}"]=analyzer.rsd
        
        analyzer.load_dem_snapshots(file_indices=list(range(250,6000,10)))
        rsd_DEM=analyzer.compute_dem_rsd(partitioner=part)
        start=(250/6000)*60
        t_DEM=np.linspace(start,60,len(rsd_DEM['rsd']))
        t_MCM=np.linspace(start,60,len(analyzer.rsd))
    
        # print(f"RSD initial: {rsd['rsd_initial']:.3f}")
        # print(f"RSD final: {rsd['rsd_final']:.3f}")
        # print(f"t_50%: {rsd['mixing_time_50']}")
        # print(f"t_90%: {rsd['mixing_time_90']}")
        # print(analyzer.concentration_history.sum(1))
        # print(analyzer.concentration_history.sum(1).shape)
        # print(analyzer.rsd)
        plt.plot(t_MCM,analyzer.rsd,"*",label="MCM")
        plt.plot(t_DEM,rsd_DEM["concentrations"].std(axis=1)/rsd_DEM["concentrations"].mean(axis=1),".",label="DEM")
        plt.title("RSD découpage 1000 partitions méthode voronoï")
        plt.xlabel("t(s)")
        plt.ylabel("RSD")
        plt.legend()
        plt.savefig('rsd.png')
        