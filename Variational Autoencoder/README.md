# Variational Autoencoders
Variational autoencoders are **Generative Models** where there task is to make a continuous latent variable space from where we can sample random points and generate new images similar to dataset. 
Simple autoencoders are used for encoding the data(say images) in some lower dimensional space where as in VAE's, a continuous distribution of latent variables is generated with given data from which we can sample datapoints in terms of latent variables and generate new images.
![encoders](https://www.renom.jp/notebooks/tutorial/generative-model/VAE/fig4.png)
As illustrated in image above, we encode **x** in **z** (generally z is lower dimensional than x) and than decode this **z** to **x_hat** in case of autoencoders. Here goal is generate **x_hat** as similar as **x**. But in case of variational autoencoder, we use our encoder network to output **mu** and **sigma**(mean and std for gaussian distribution of different latent variable dimensions.
