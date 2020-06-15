import numpy as np
from skimage.morphology import skeletonize_3d
from network_3d import skel2graph, plot_graph
from mayavi import mlab

#Create example data, the letters A,B and X overlaping in space
from PIL import Image, ImageDraw, ImageFont

font = ImageFont.truetype("verdana.ttf", 65)
bin=np.zeros((150,150,25),dtype=bool)
image = Image.new("L", (150,150))
draw = ImageDraw.Draw(image)


draw.text((0,0), 'A', (1), font=font)
bin[:,:,1:11]=np.array(image)[:,:,np.newaxis]
image = Image.new("L", (150,150))
draw = ImageDraw.Draw(image)
draw.text((30,30), 'B', (1), font=font)
bin[:,:,7:17]= np.logical_or(bin[:,:,7:17],np.array(image)[:,:,np.newaxis])
image = Image.new("L", (150,150))
draw = ImageDraw.Draw(image)
draw.text((60,60), 'X', (1), font=font)
bin[:,:,14:24]=np.logical_or(bin[:,:,14:24],np.array(image)[:,:,np.newaxis])



skel = skeletonize_3d(bin) #scikit-image function
skel = skel.astype(np.bool) #data needs to be bool
G = skel2graph(skel) #create graph

#plot the graph, use the z component to colorcode both the edges and the nodes, scale nodes according to their degree
plot_graph(G,node_color_keyword='z',edge_color_keyword='z',scale_node_keyword='degree')

#show the binary data
mlab.contour3d(bin.astype(np.float),contours=[.5],opacity=.5,color=(1,1,1))

mlab.show()

