# Find your babies in the latent space of StyleGAN - Runway ML port

Upload and analyze two peoples faces to generate an image of a speculative Baby.
Works by mixing two peoples latent vectors and then do age transformation using StyleGan. 

Based on [http://github.com/Puzer/stylegan-encoder] & [https://colab.research.google.com/drive/139OhnW0O_3-4IrnUCXRkO9nJn38qcdsi]


## How to install

Clone this repo into your Github account and then import to Runway.

See how this works here [https://learn.runwayml.com/#/how-to/import-models]


## How to use
There are two sections
* Encode
* Generate

Switch between those two using the "Command" dropdown menu.

### Encode
In this section an encoder will search for your uploaded faces in the latent space of StyleGAN.
You need to upload an image of somebodies face using Runways UI (File input)

The image has to be **exactly 512x512** pixels in size!

Use the iteration slider to set the amount of time the model will spend on analyzing your image.
**The process will start as soon as you move the slider or change the number in the box.
Right now there is a problem with running more than 150 iterations. So please stick below.**
If you want to analyze another face upload another 512x512 image and change the amount of iterations to start.


### Generate
In this section one or two faces will be mixed and then transformed to an image of a new generated person.
Results of the Encode section can be selected in the "Person 1" or "Person 2" section of the UI.
Use the sliders to tune the results.




