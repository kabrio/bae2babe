# https://colab.research.google.com/drive/1RUaVUqCvyojwoMglp6cFoLDnCfLHBZtB#scrollTo=PVYrjvgE_8AU

import helpers
import os
import argparse
import pickle
import PIL.Image
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import config
from encoder.generator_model import Generator
from encoder.perceptual_model import PerceptualModel
import matplotlib.pyplot as plt
from tqdm import tqdm
import runway

preLoad1 = np.load("ffhq_dataset/latent_representations/hillary_clinton_01.npy")
preLoad2 = np.load("ffhq_dataset/latent_representations/donald_trump_01.npy")
prevIterations = -1
generated_dlatents = 0
generator = 0

blank_img = PIL.Image.new('RGB', (512, 512), (127, 0, 127))


@runway.setup(options={'checkpoint': runway.file(extension='.pkl')})
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
	perceptual_model = PerceptualModel(opts[512], layer=9, batch_size=1)
	perceptual_model.build_perceptual_model(generator.generated_image)
	return generator

# def setup(opts):
# 	tflib.init_tf()
# 	url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
# 	with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
# 		_G, _D, Gs = pickle.load(f)
# 		# _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
# 		# _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
# 		# Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.
# 	Gs.print_layers()
# 	global generator
# 	generator = Generator(Gs, batch_size=1, randomize_noise=False)
# 	return Gs

def generate_image(generator, latent_vector):
	latent_vector = latent_vector.reshape((1, 18, 512))
	generator.set_dlatents(latent_vector)
	img_array = generator.generate_images()[0]
	img = PIL.Image.fromarray(img_array, 'RGB')
	return img.resize((512, 512))   


# ENCODING

generate_inputs_1 = {
	'portrait': runway.image(),
	'iterations': runway.number(min=1, max=5000, default=10, step=1.0),
	'encode': runway.boolean(default=False)
}
generate_outputs_1 = {
	'image': runway.image(width=512, height=512)
}

encodeCount = 0
latent_vectors = []
latent_vectors.append(preLoad1)
latent_vectors.append(preLoad2)
latent_vectors.append(preLoad1)
latent_vectors.append(preLoad2)

@runway.command('encode', inputs=generate_inputs_1, outputs=generate_outputs_1)
def find_in_space(model, inputs):
	global generated_dlatents
	global prevIterations
	global encodeCount
	global blank_img
	image = blank_img
	if (inputs['iterations'] != prevIterations and inputs['encode']):
		prevIterations = inputs['iterations']
		if (encodeCount > 3):
			encodeCount = 0
		generator.reset_dlatents()
		names = ["looking at you!"]
		perceptual_model.set_reference_images(inputs['portrait'])
		print ("image loaded.")
		print ("encoding #", encodeCount+1, " for ", prevIterations, " iterations.")
		op = perceptual_model.optimize(generator.dlatent_variable, iterations=inputs['iterations'], learning_rate=1.)
		# load latent vectors	
		generated_dlatents = generator.get_dlatents()
		pbar = tqdm(op, leave=False, total=inputs['iterations'], mininterval=2.0, miniters=10, disable=True)
		for loss in pbar:
			pbar.set_description(' '.join(names)+' Loss: %.2f' % loss)
		print(' '.join(names), ' loss:', loss)
		print ("finished encoding: ", encodeCount+1) 		
		# Generate images from found dlatents
		print ("generating image: ", encodeCount+1)
		latent_vectors[encodeCount] = generator.get_dlatents()
		image = generate_image(generator, latent_vectors[encodeCount])
		# generator.set_dlatents(latent_vectors[encodeCount])
		# generated_images = generator.generate_images()
		# for img_array, dlatent, img_name in zip(generated_images, generated_dlatents, names):
		# 	img = PIL.Image.fromarray(img_array, 'RGB')
		# 	img.resize((512, 512))
		
		# print(latent_vectors)
		inputs['encode'] = False
		encodeCount += 1
	else:
		print("Did not encode.")		

	return{"image": image}

# GENERATION

cat_1 = runway.category(choices=["1", "2", "3", "4"], default="1")
cat_2 = runway.category(choices=["1", "2", "3", "4"], default="1")


generate_inputs_2 = {
	'person_1': cat_1,
	'person_2': cat_2,
	'mix': runway.number(min=0.0, max=100.0, default=50.0, step=1.0),
	'age': runway.number(min=-10.0, max=10.0, default=5.0, step=0.1),
	'fine_age': runway.number(min=-1.0, max=1.0, default=0.0, step=0.01),
	'smile': runway.number(min=-10.0, max=10.0, default=0, step=0.1),
	'gender': runway.number(min=-10.0, max=10.0, default=0, step=0.1)
}
generate_outputs_2 = {
	'image': runway.image(width=512, height=512)
}

@runway.command('generate', inputs=generate_inputs_2, outputs=generate_outputs_2)
def move_and_show(model, inputs):
	print("mixing")
	global latent_vector_2
	global latent_vector_1
	#latent_vector_1 = np.load("latent_representations/hee.npy")
	latent_vector_1 = latent_vectors[int(inputs['person_1'])-1].copy()
	latent_vector_2 = latent_vectors[int(inputs['person_2'])-1].copy()
	latent_vector = (latent_vector_2 * (inputs['mix']/100) + latent_vector_1 * (1.0 - inputs['mix']/100)) / 2
	# latent_vector = latent_vectors[int(inputs['person 1'])-1].copy()

	# load direction
	age_dir = np.load('ffhq_dataset/latent_directions/age.npy')
	smile_dir = np.load('ffhq_dataset/latent_directions/smile.npy')
	gender_dir = np.load('ffhq_dataset/latent_directions/gender.npy')
	# model = generator
	coeff = inputs['age'] + inputs['fine_age']
	smile_c = inputs['smile']
	gender_c = inputs['gender']
	new_latent_vector = latent_vector.copy()
	new_latent_vector[:8] = (latent_vector + coeff*age_dir + smile_c*smile_dir + gender_c*gender_dir)[:8]
	image = (generate_image(model, new_latent_vector))
	#ax[i].set_title('Coeff: %0.1f' % coeff)
	#plt.show()
	return {'image': image}

if __name__ == '__main__':
	runway.run()