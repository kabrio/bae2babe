# https://colab.research.google.com/drive/1RUaVUqCvyojwoMglp6cFoLDnCfLHBZtB#scrollTo=PVYrjvgE_8AU

import helpers
import os
import argparse
import pickle
from tqdm import tqdm
import PIL.Image
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import config
import matplotlib.pyplot as plt
from encoder.generator_model import Generator
from encoder.perceptual_model import PerceptualModel
import runway

prevIterations = -1
generated_dlatents = 0

@runway.setup(options={'checkpoint': runway.file(extension='.pkl'), 'image dimensions': runway.number(min=128, max=1024, default=512, step=128)})
def setup(opts):
	# Initialize generator and perceptual model
	global perceptual_model
	global generator
	tflib.init_tf()
	model = opts['checkpoint']
	print("open model %s" % model)
	with open(model, 'rb') as file:
		G, D, Gs = pickle.load(file)
	Gs.print_layers()	
	generator = Generator(Gs, batch_size=1, randomize_noise=False)		
	perceptual_model = PerceptualModel(opts['image dimensions'], layer=9, batch_size=1)
	perceptual_model.build_perceptual_model(generator.generated_image)
	return Gs


def generate_image(generator, latent_vector):
	latent_vector = latent_vector.reshape((1, 18, 512))
	generator.set_dlatents(latent_vector)
	img_array = generator.generate_images()[0]
	img = PIL.Image.fromarray(img_array, 'RGB')
	return img.resize((512, 512))   

generate_inputs = {
	'portrait': runway.image(),
	'iterations': runway.number(min=1, max=5000, default=10, step=1.0),
	'age': runway.number(min=-30, max=20, default=4, step=0.2)
}

# generate_outputs = {
# 	'latent_vector': runway.file()
# }

generate_outputs = {
	'generated': runway.image(width=512, height=512)
	# 'vector': runway.vector(length=512)
	# "hextext": runway.text
}

@runway.command('encode', inputs=generate_inputs, outputs=generate_outputs)
def find_in_space(model, inputs):
	global generated_dlatents
	global prevIterations
	if (inputs['iterations'] != prevIterations):
		generator.reset_dlatents()
		names = ["looking at you!"]
		perceptual_model.set_reference_images(inputs['portrait'])
		print ("image loaded.")
		prevIterations = inputs['iterations']
		print ("encoding for: ", prevIterations, " iterations.")
		op = perceptual_model.optimize(generator.dlatent_variable, iterations=inputs['iterations'], learning_rate=1.)
		# load latent vectors	
		generated_dlatents = generator.get_dlatents()
		pbar = tqdm(op, leave=False, total=inputs['iterations'], mininterval=2.0, miniters=10, disable=True)
		for loss in pbar:
			pbar.set_description(' '.join(names)+' Loss: %.2f' % loss)
		print(' '.join(names), ' loss:', loss)
	else:
		names = ["mixing you."]

	print ("mixing new human")	

	# modify latent vectors
	# load direction
	age_direction = np.load('ffhq_dataset/latent_directions/age.npy')
	direction = age_direction
	# model = generator
	coeff = inputs['age']
	new_latent_vector = generated_dlatents.copy()
	new_latent_vector[:8] = (new_latent_vector + coeff*direction)[:8]

	# Generate images from found dlatents and save them
	generator.set_dlatents(new_latent_vector)
	generated_images = generator.generate_images()
	for img_array, dlatent, img_name in zip(generated_images, generated_dlatents, names):
		img = PIL.Image.fromarray(img_array, 'RGB')
		img.resize((512, 512))
	#	img.save(os.path.join(args.generated_images_dir, f'{img_name}.png'), 'PNG')
	#	np.save(os.path.join(args.dlatent_dir, f'{img_name}.npy'), dlatent)

	#return {'vector': generated_dlatents}
	print ("returning image")
	return {'generated': img}
	generated_dlatents = generator.get_dlatents()
	s = generated_dlatents.tobytes()
	s2 = s.hex()
	#print ("returning text")
	#return{"hextext": s2}

if __name__ == '__main__':
	runway.run(debug=True)