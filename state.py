"""
saves variables for different pages of web_train

"""

#loss.py:
loss_name = 'Dice Loss'
jac = False
sq = True
##network.py:
dropout= 0.0
smallest_layer = 16
largest_layer = 128

#input
size =512
image_dir = None
channels = 3
top_hat = False
lbp = False

# spatial transforms:
flip = False
spatial_axis = None
p_flip = 0.5

rotate= False
range_x = 0.0
range_y =0.0
p_rotate = 0.5

croping = False
roi_scale= 0.8
max_roi_scale = 0.9

zoom=False
min_zoom = 0.9
max_zoom = 1.1
p_zoom = 0.5


#intensity transforms:
gauss_noise = False
gaus_prop = 0.5

one_off = True

gibbs_noise = False
p_gibs = 0.5



RandHistogramShift = False
hist_prop = 0.5
control_points =10

RandCoarseDropout = False
coarse_prop = 0.5
holes = 20
hole_size = 10