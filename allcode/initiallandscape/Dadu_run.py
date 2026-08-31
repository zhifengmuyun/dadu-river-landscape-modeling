from badlands.model import Model as badlandsModel

# initialise model
model = badlandsModel()

model.load_xml('base.xml')
model.run_to_time(5000000)

