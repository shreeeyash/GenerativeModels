# Variational Autoencoders
Generative Models are subset of machine learning where our task is to make a model which can *generate images*, not same but similar to our dataset on which model is trained. VAE is one of the most successful approch towards the problem of generating images similar to dataset.

Variational autoencoders are **Generative Models** where there task is to make a continuous latent variable space from where we can sample random points and generate new images similar to dataset. 
Simple autoencoders are used for encoding the data(say images) in some lower dimensional space where as in VAE's, a continuous distribution of latent variables is generated with given data from which we can sample datapoints in terms of latent variables and generate new images.
![encoders](https://www.renom.jp/notebooks/tutorial/generative-model/VAE/fig4.png)
As illustrated in image above, we encode **x** in **z** (generally z is lower dimensional than x) and than decode this **z** to **x'** in case of autoencoders. Here goal is generate **x'** as similar as **x**. But in case of variational autoencoder, we use our encoder network to output **mu** and **sigma**(mean and std for gaussian distributions of different latent variable dimensions). After having generated distribution parameters of latent variables, we sample random points from generated distribution and then decoder model comes into action. The decoder network constructs an image for given latent variable, after which we can use loss function to optimize the parameters.</br>

### Difference between Variational Autoencoder(VAE) and Generative Adversarial Networks(GAN)

The basic difference difference comes in terms of *Explicit & Implicit density functions*. GAN works on principle of game theory. The two players are Generator and Discriminator networks. Generator is a deep neural network whose task is to generate images that are indistinguishable from the images sampled from real dataset. whereas discriminator tries to panalize fake images(that are coming from generator network rather than original dataset).
![GAN](https://skymind.ai/images/wiki/GANs.png)
As described, task in GAN is not to pridict any probability distribution. Rather the task is to sample images from generative network that are similar to images in dataset. Hence here our model is learning an **Implicit Density Function** in terms of parameters of generative network.</p>
Task of variational autoencoder is somewhat different. VAE take the real image from dataset as input and tries to find the hidden probability density function in terms of mean and variance(assuming latent code to be gaussian distributed). Hence in VAE we are predicting an **Explicit Density Function** in terms of mean and variance as shown in figure below.
![VAE](https://i.imgur.com/ZN6MyTx.png)

### Problem of Intractability

To visualize the problem of intractability, we may use distribution of digits in MNIST dataset as shown below.
<img src="https://cdn-images-1.medium.com/max/975/1*-i8cp3ry4XS-05OWPAJLPg.png" width="500">
As we want our model to be able to generate images that lie somewhere inbetween different classes of digits, we may face problem of Intractability. If the space has discontinuities (eg. gaps between clusters) and you sample/generate a variation from there, the decoder will simply generate an unrealistic output, because the decoder has no idea how to deal with that region of the latent space.  


