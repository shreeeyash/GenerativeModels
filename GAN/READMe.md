# Generative adversarial network
####  Problems with learning GAN
The main problem during training of GAN arises due to awkward value of divergence(KL-divergence). When dimension of *P*r and *P*g is less than union, the KL-divergence will explode to infinity. The two distribution are separated in union space. In this case KL-divergence is a bad numeric which cant help the generator to learn better. The onle case when GAN can learn successfully is that the two distributions are intersecting.</br>
![](https://cdn-images-1.medium.com/max/1300/1*xRjphX2OGhfDllYFIkabzw.png)</br>
