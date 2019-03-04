# Generative adversarial network
### Used Model 
</br> ![](https://github.com/Shreeyash-iitr/GenerativeModels/blob/master/GAN/dcgan.png) </br>
Although numbers of channels in each layer(except last one) is reduced to half due to computation limit.

###  Problems with learning GAN
The main problem during training of GAN arises due to awkward value of divergence(KL-divergence). When dimension of *P*r and *P*g is less than union, the KL-divergence will explode to infinity. The two distribution are separated in union space. In this case KL-divergence is a bad numeric which cant help the generator to learn better. The onle case when GAN can learn successfully is that the two distributions are intersecting.</br>
![](https://cdn-images-1.medium.com/max/1300/1*xRjphX2OGhfDllYFIkabzw.png)</br>

### Experiments & Results 
I used Pokemon dataset having 4200 images of different pokemons for training DC-GAN. SGD with momentum is used for discriminator and Adam optimizer is used for generator network. Learning rate was set to 0.0002 as mentioned in DC-GAN paper.Th e following gif shows images after each 5 epochs of training. Model was trained for 125 epochs. Final images look similar to newly created pokemons but are not clear due to lesser amount of data available.</br>
![](https://github.com/Shreeyash-iitr/GenerativeModels/blob/master/GAN/results/pokemon.gif)</br>

#### Last Epoch Generated Image

![](https://github.com/Shreeyash-iitr/GenerativeModels/blob/master/GAN/results/Pokemon%20Generated%20Images/individualImage.png)</br>
