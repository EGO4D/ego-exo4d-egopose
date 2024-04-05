import trimesh 

from libzhifan import geometry
from libzhifan.geometry import SimpleMesh

verts, faces = geometry.canonical_cuboids(
    x=0, y=0, z=3,
    w=2, h=2, d=2,
    convention='opencv',
    return_mesh=False
)


mesh = SimpleMesh(verts, faces)
scene = trimesh.scene.Scene([mesh])
a = mesh.synced_mesh

# pcd = SimplePCD(verts)
# scene = trimesh.scene.Scene([pcd])