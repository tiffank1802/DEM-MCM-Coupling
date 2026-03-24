import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches

class SimpleDEM: 
    def __init__(self,n_particles, radius=0.01,mass=0.001):
        self.pos=np.random.randn(n_particles,3)*.1
        self.pos[:,2]=np.abs(self.pos[:,2])+0.5
        self.pos_init=self.pos.copy()
        self.vel=np.zeros((n_particles,3))
        self.radius=radius
        self.mass=mass
        self.k=2000 # Raideur de contact
        self.dt=1e-4
        self.damping=.95
        
        # Historique des trajectoires
        self.history=[self.pos.copy()]
        self.energy_history=[]

        
    def detect_collision(self):
        # Détection naive o(n2) - Optimisable acex grille spatiale
        for i in range(len(self.pos)):
            for j in range(i+1, len(self.pos)):
                dist=np.linalg.norm(self.pos[i]-self.pos[j])
                if dist<2*self.radius:
                    yield i,j,dist 
    def compute_forces(self):
        forces=np.zeros_like(self.pos)
        forces[:,2]-=9.81*self.mass
        
        for i,j,dist in self.detect_collision():
            # Force de contact ressort
            overleap=2*self.radius-dist
            direction=(self.pos[i]-self.pos[j])/dist
            f=self.k*overleap*direction
            
            forces[i]+=f
            forces[j]-=f
        return forces
    def step(self):
        """Integration Euler semi-explicite"""
        forces=self.compute_forces()
        self.vel+=forces/self.mass*self.dt
        self.vel*=self.damping # Amortissement global
        self.pos+=self.vel*self.dt
        
        
        # conditions aux limites (sol)
        mask=self.pos[:,2]<self.radius
        self.pos[mask,2]=self.radius
        self.vel[mask,2]*=-0.6
        
        # Murs latéraux (boîte)
        for dim in [0,1]:
            mask_low=self.pos[:,dim]<-0.5
            mask_high=self.pos[:,dim]>0.5
            self.pos[mask_low,dim]=-.5
            self.pos[mask_high,dim]=.5
            self.vel[mask_high|mask_low,dim]*=-.6
            
        # Historique
        if len(self.history)<500:
            self.history.append(self.pos.copy())
        
        # Energie
        ke=.5*self.mass*np.sum(self.vel**2)
        pe=self.mass*9.81*np.sum(self.pos[:,2])
        self.energy_history.append([ke,pe,ke+pe])
        return self.pos.copy()


class DEMVisualizer:
    """Class for Visualization and animations
    """

    def __init__(self,sim):
        self.sim=sim
        self.fig=plt.figure(figsize=(14,10))
        
        # Vue 3D principale
        self.ax_3d=self.fig.add_subplot(2,2,1,projection="3d")
        self.ax_3d.set_ylim(-0.6,0.6)
        self.ax_3d.set_zlim(0,1.0)
        self.ax_3d.set_xlabel('X')
        self.ax_3d.set_ylabel('Y')
        self.ax_3d.set_label('Z')
        self.ax_3d.set_title('DEM 3D Temps Réel')
        
        # Vue de dessus (X-Y)
        self.ax_top=self.fig.add_subplot(2,2,2)
        self.ax_top.set_xlim(-.6,.6)
        self.ax_top.set_ylim(-.6,.6)
        self.ax_top.set_xlabel('X')
        self.ax_top.set_ylabel('Y')
        self.ax_top.set_title('Vue de dessus')
        self.ax_top.set_aspect('equal')
        
        # Vue de côté (X-Z)
        self.ax_side=self.fig.add_subplot(2,2,3)
        self.ax_side.set_xlim(-.6,.6)
        self.ax_side.set_ylim(0,1.0)
        self.ax_side.set_xlabel('X')
        self.ax_side.set_ylabel('Z')
        self.ax_side.set_title('Vue de côté')
        self.ax_side.set_aspect('equal')
        
        # Énergie
        self.ax_energy=self.fig.add_subplot(2,2,4)
        self.ax_energy.set_xlabel('Step')
        self.ax_energy.set_ylabel('Energie [J]')
        self.ax_energy.set_title('Energie cinetique/potentielle/totale')
        self.ax_energy.grid(True)
        
        # Initialisation des plots
        self.scatter_3d=None
        self.scatter_top=None
        self.scatter_side=None
        self.lines_energy=[]
        
        plt.tight_layout()
        plt.suptitle('Simulation DEM - Animation Temps Réel', y=1.02)
    
    def init_animation(self):
        """Initialisation des éléments grahiques"""
        self.scatter_3d=self.ax_3d.scatter([],[],[],c='blue',s=50,alpha=.6)
        
        # 2D 
        self.scatter_top=self.ax_top.scatter([],[],c='red',s=30,alpha=.6)
        self.scatter_side=self.ax_side.scatter([],[],c='green',s=30,alpha=.6)
        
        # Energie
        self.lines_energy=[
            self.ax_energy.plot([],[],'r-',label='Cinetique',alpha=.7)[0],
            self.ax_energy.plot([],[],'r-',label='Potentielle',alpha=.7)[0],
            self.ax_energy.plot([],[],'r-',label='Totale',alpha=.7,linewidth=2)[0],
            
        ]
        self.ax_energy.legend()
        
        return[self.scatter_3d,self.scatter_top,self.scatter_side]+self.lines_energy
    
    def update(self,frame):
        """Met à jour l'animation

        Args:
            frame (int): _description_
        """            """"""
        # Avance simulatoin
        for _ in range(5):
            self.sim.step()
            pos=self.sim.pos.copy()
            
            # Mise à jour 3D
            self.scatter_3d._offse3d=(pos[:,0],pos[:,1],pos[:,2])
            
            # couleurs selon vitesse
            speeds=np.linalg.norm(self.sim.vel,axis=1)
            colors=plt.cm.viridis(speeds/(np.max(speeds)+.001))
            self.scatter_3d.set_color(colors)
            
            # Mise à jour 2D
            self.scatter_top.set_offsets(pos[:,:2])
            self.scatter_side.set_offsets(pos[:,[0,2]])
            
            # Mise à jour de l'énergie
            
            if (len(self.sim.energy_history)>0):
                energy=np.array(self.sim.energy_history)
                steps=np.arange(len(energy))
                for i, line in enumerate(self.lines_energy):
                    line.set_data(steps,energy[:,i])
                self.ax_energy.set_xlim(0,max(100,len(energy)))
                self.ax_energy.set_ylim(0,np.max(energy)*1.1)
                
                
            #Titre avec Info
            ke=0.5*self.sim.mass*np.sum(self.sim.vel**2)
            self.ax_energy.set_title(f"DEM 3D - Step {frame*5} -KE={ke:.4f} J")
            return[self.scatter_3d,self.scatter_top,self.scatter_side]+self.lines_energy
        
    def run(self,frames=500,interval=50):
        """Lance L'animation

        Args:
            frames (int, optional): _description_. Defaults to 500.
            interval (int, optional): _description_. Defaults to 50.
        """
        print("Démarrage animation ...")
        print("Fermez la fenêtre pour l'arrêter")
        
        anim=FuncAnimation(
            self.fig,
            self.update,
            init_func=self.init_animation,
            frames=frames,
            interval=interval,
            blit=False,
            repeat=False
        )
        
        # plt.savefig("visuals.png")
        anim.save("dem_simulation.gif",writer='ffmpeg',fps=30,dpi=150)
# sim=SimpleDEM(n_particles=50,radius=.03,mass=.01)

if __name__=="__main__":
    
# Utilisation
    print("Initialisation DEM ...")
    sim=SimpleDEM(n_particles=50,radius=.03,mass=.01)
    viz=DEMVisualizer(sim=sim)
    animation=viz.run(frames=1000, interval=30)
    
    print("Animation terminée")
    
# np.savetxt("positions.txt",sim.pos)
