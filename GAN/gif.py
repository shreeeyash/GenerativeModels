import imageio
images = []
for e in range(5, 125, 5):
    img_name = '/home/shreeyash/Desktop/Deep Learning/Generative Models/GAN/results/Pokemon Generated Images/'+str(e) + '.png'
    images.append(imageio.imread(img_name))
imageio.mimsave(
    '/home/shreeyash/Desktop/Deep Learning/Generative Models/GAN/results/pokemon.gif', images, fps=2)
