from mujoco import MjSim, MjRenderContextOffscreen
import mujoco.viewer
import mujoco
import time

model = mujoco.MjModel.from_xml_path("your_model.xml")
sim = MjSim(model)

start = time.time()
ctx = MjRenderContextOffscreen(sim.model, sim.data)
sim.render()
print("Rendering initialized in", time.time() - start, "seconds")
