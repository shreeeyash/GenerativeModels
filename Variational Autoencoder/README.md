# Variational Autoencoders
Generative Models are subset of machine learning where our task is to make a model which can *generate images*, not same but similar to our dataset on which model is trained. VAE is one of the most successful approch towards the problem of generating images similar to dataset.

Variational autoencoders are **Generative Models** where there task is to make a continuous latent variable space from where we can sample random points and generate new images similar to dataset. 
Simple autoencoders are used for encoding the data(say images) in some lower dimensional space where as in VAE's, a continuous distribution of latent variables is generated with given data from which we can sample datapoints in terms of latent variables and generate new images.
![encoders](https://www.renom.jp/notebooks/tutorial/generative-model/VAE/fig4.png)
As illustrated in image above, we encode **x** in **z** (generally z is lower dimensional than x) and than decode this **z** to **x'** in case of autoencoders. Here goal is generate **x'** as similar as **x**. But in case of variational autoencoder, we use our encoder network to output **mu** and **sigma**(mean and std for gaussian distributions of different latent variable dimensions). After having generated distribution parameters of latent variables, we sample random points from generated distribution and then decoder model comes into action. The decoder network constructs an image for given latent variable, after which we can use loss function to optimize the parameters.</br>

### Difference between Variational Autoencoder(VAE) and Generative Adversarial Networks(GAN)

The basic difference difference comes in terms of *Explicit & Implicit density functions*.
