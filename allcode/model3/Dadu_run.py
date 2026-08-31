from badlands.model import Model as badlandsModel

# initialise model
model = badlandsModel()

model.load_xml('model3.xml')
model.run_to_time(5000000)

