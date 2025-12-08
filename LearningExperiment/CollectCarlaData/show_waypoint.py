import carla

client = carla.Client('localhost', 2000)
world = client.load_world('Town03')
vehicles = world.get_actors().filter('vehicle.*')
client.apply_batch([carla.command.DestroyActor(vehicle) for vehicle in vehicles])

m = world.get_map()
transform = carla.Transform()
spectator = world.get_spectator()

bv_transform = carla.Transform(transform.location + carla.Location(z=250, x=0), carla.Rotation(yaw=0, pitch=-90))

spectator.set_transform(bv_transform)

blueprint_library = world.get_blueprint_library()
spawn_points = m.get_spawn_points()

for i, spawn_point in enumerate(spawn_points):
    world.debug.draw_string(spawn_point.location, str(i), life_time=100)
    world.debug.draw_arrow(spawn_point.location, spawn_point.location + spawn_point.get_forward_vector(), life_time=100)

while True:
    world.tick()