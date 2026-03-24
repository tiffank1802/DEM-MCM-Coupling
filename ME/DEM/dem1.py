from yade import utils,pack,geom,qt

# création du matériau
O.materials.append(FrictMat(young=1e7,poisson=.3,frictionAngele=.5))

# Géneration des sphères
spheres=pack.randomDensePack(
    predicate=pack.inSphere((O,0,0),0.5),
    raduis=0.01,
    spheresIncel=1000
)

#ajout au moteur
O.bodies.append(spheres)

#Definition du moteur de simulation

O.engines=[
    ForceResetter(),
    InsertionSortCollider([Bol_Sphere_Aabb()]),
    InteractionLoop(
    [Ig2_Sphere_ScGeom()],
    [Ip2_FrictMat_FrictMat_FrictPhys()],
    [Law2_scGeom_FrictPhys_CundallStrack()]
    ),
    NextonIntegrator(gravity=(0,0,-9.81)),
]
O.run()
