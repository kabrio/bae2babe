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


generate_inputs = {
	'portrait': runway.image(),
	'iterations': runway.number(min=1, max=5000, default=10, step=1.0)
}

# generate_outputs = {
# 	'latent_vector': runway.file()
# }

generate_outputs = {
	'generated': runway.image(width=512, height=512)
}

@runway.command('encode', inputs=generate_inputs, outputs=generate_outputs)
def find_in_space(model, inputs):
	names = ["looking at you!"]
	perceptual_model.set_reference_images(inputs['portrait'])
	print ("image loaded")
	print ("encoding for: ", inputs['iterations'])
	op = perceptual_model.optimize(generator.dlatent_variable, iterations=inputs['iterations'], learning_rate=1.)
	pbar = tqdm(op, leave=False, total=inputs['iterations'], mininterval=30.0, miniters=50)
	for loss in pbar:
		pbar.set_description(' '.join(names)+' Loss: %.2f' % loss)
	print(' '.join(names), ' loss:', loss)

	# Generate images from found dlatents and save them
	generated_images = generator.generate_images()
	generated_dlatents = generator.get_dlatents()
	for img_array, dlatent, img_name in zip(generated_images, generated_dlatents, names):
		img = PIL.Image.fromarray(img_array, 'RGB')
		img.resize((512, 512))
	#	img.save(os.path.join(args.generated_images_dir, f'{img_name}.png'), 'PNG')
	#	np.save(os.path.join(args.dlatent_dir, f'{img_name}.npy'), dlatent)

	generator.reset_dlatents()
	print ("returning generated image")
	#return {'latent_vector' : dlatent}
	return {'generated': img}

if __name__ == '__main__':
	runway.run(debug=True)