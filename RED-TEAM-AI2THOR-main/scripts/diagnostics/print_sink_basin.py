import ai2thor.controller

controller = ai2thor.controller.Controller(scene="FloorPlan1")

# initial
print("Before picking up pot:")
for obj in controller.last_event.metadata['objects']:
    if "Sink" in obj['objectType'] or "Sink" in obj['objectId']:
        print(obj['objectId'])

# pick up pot
pot_id = next(o['objectId'] for o in controller.last_event.metadata['objects'] if o['objectType'] == "Pot")
controller.step(action="PickupObject", objectId=pot_id)

print("\nAfter picking up pot:")
for obj in controller.last_event.metadata['objects']:
    if "Sink" in obj['objectType'] or "Sink" in obj['objectId']:
        print(obj['objectId'])
