import numpy as np
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm
from argparse import ArgumentParser


####DEFINE ALL FUNCTIONS####

# create proportions of canvas, making sure that the height/width ratio is reasonable
# (default reasonable ratio is golden ratio phi=1.62)
def create_canvas(lower_bound=500, upper_bound=1500, ratio=1.62):
    
    height = int(np.random.uniform(lower_bound, upper_bound))
    #make sure that width stays within proportion of height
    width = int(np.random.uniform(height/ratio, height*ratio))

    # while True:
    #   width = int(np.random.uniform(lower_bound/ratio, upper_bound*ratio))
    #   if width < (height / ratio) or width > (height * ratio):
    #       break

    canvas = np.ones((width, height, 3))
    canvas[0, :, :] = 0
    canvas[width-1,:,:] = 0
    canvas[:,0,:] = 0
    canvas[:,height-1,:] = 0
    
    return canvas, width, height

# define horizontal line thickness and location, and draw it onto canvas
def draw_a_line_h(canvas, width, height):
    
    line_width = int(width/np.random.uniform(50,70))+1
    where_line_drawn = int(np.random.uniform(height/10, 9*height/10))

    canvas[where_line_drawn:where_line_drawn+line_width, :] = 0
    
    return canvas, line_width, where_line_drawn

# define vertical line thickness and location, and draw it onto canvas
def draw_a_line_v(canvas, width, height):
    
    line_width = int(height/np.random.uniform(50,70))+1
    where_line_drawn = int(np.random.uniform(height/10, 9*height/10))

    canvas[:, where_line_drawn:where_line_drawn+line_width] = 0
    
    return canvas, line_width, where_line_drawn


# draw horizontal and vertical lines onto canvas, store the location and widths of lines
def draw_lines(canvas, width, height, number_of_vertical_lines=7, number_of_horizontal_lines=7):
    
    h_line_widths_locs = np.zeros((number_of_horizontal_lines,2))
    v_line_widths_locs = np.zeros((number_of_vertical_lines,2))
    

    for i in range(number_of_vertical_lines):
        canvas, v_line_widths_locs[i,0], v_line_widths_locs[i,1] = draw_a_line_v(canvas, width, height)
    for i in range(number_of_horizontal_lines):
        canvas, h_line_widths_locs[i,0], h_line_widths_locs[i,1] = draw_a_line_h(canvas, width, height)
        
    return canvas, h_line_widths_locs, v_line_widths_locs
    
# find the x-y location of the top left and bottom right corners of each patch of painting, return 2 arrays
def location_corners(canvas, width, height, number_of_corners):
    
    #define patch of canvas that constitutes a top left and bottom right corner
    #topleft looks like
    #[0 0]
    #[0 1]
    #bottomright looks like
    #[1 0]
    #[0 0]
    topleft = np.copy(canvas[0:2,0:2,:])
    bottomright = np.copy(canvas[width-2:width, height-2:height,:])
    
    #the code crashes if this is less than 150000... I should probably find the bug at some point, but
    #it works for now
    toplefts = np.zeros((number_of_corners*150000, 2))
    bottomrights = np.zeros((number_of_corners*150000, 2))

    iters_topleft = 0
    iters_bottomright = 0

    for i in range(width-1):
        for j in range(height-1):
            compare_topleft = np.array_equal(canvas[i:i+2,j:j+2,:], topleft)
            compare_bottomright = np.array_equal(canvas[i:i+2, j:j+2,:], bottomright)
            if compare_topleft:
                toplefts[iters_topleft,0]=i
                toplefts[iters_topleft,1]=j
                iters_topleft += 1
            if compare_bottomright:
                bottomrights[iters_bottomright,0]=i
                bottomrights[iters_bottomright,1]=j
                iters_bottomright += 1 
                
    return toplefts, bottomrights

def location_corners_better(horizontals, verticals, width, height):

    #define locations for topleft
    h_tl = np.sort(horizontals[:,0]+horizontals[:,1]-1)
    v_tl = np.sort(verticals[:,0]+verticals[:,1]-1)
    h_tl = np.insert(h_tl, 0, [0])
    v_tl = np.insert(v_tl, 0, [0])
    
    #define locations for bottomright
    h_br = np.sort(np.copy(horizontals[:,1])-1)
    v_br = np.sort(np.copy(verticals[:,1])-1)
    h_br = np.append(h_br, width)
    v_br = np.append(v_br, height)
    
    num_corners = len(h_tl)*len(v_tl)
    toplefts = np.zeros((num_corners,2))
    bottomrights = np.zeros((num_corners,2))

    for i, r in enumerate(itertools.product(h_tl, v_tl)):
        toplefts[i] = [r[0],r[1]]
        
    for i, r in enumerate(itertools.product(h_br, v_br)):
        bottomrights[i] = [r[0],r[1]]
    
        
    return toplefts, bottomrights

def choose_corners_normal(toplefts, bottomrights):
    
    toplefts = toplefts.astype(int)
    bottomrights = bottomrights.astype(int)
    
    corners = np.concatenate((toplefts, bottomrights), axis=1)
    np.random.shuffle(corners)
    
    return corners

#permutes corners so that they're not all of the same rectangle
def choose_corners_funky(toplefts, bottomrights):

    toplefts = toplefts.astype(int)
    bottomrights = bottomrights.astype(int)

    
    np.random.shuffle(bottomrights)
    np.random.shuffle(toplefts)
    funky_corners = np.concatenate((toplefts, bottomrights), axis=1)
    
    return funky_corners

def populate_colours(canvas, corners, how_many_patches=15):
    
    fresh_canvas = np.copy(canvas)
    
    if len(corners) < how_many_patches:
        how_many_patches = len(corners)

    for i in range(how_many_patches):
        x_topleft = corners[i,0]
        y_topleft = corners[i,1]
        x_bottomrights = corners[i,2]
        y_bottomrights = corners[i,3]
        fresh_canvas[x_topleft+1:x_bottomrights+1,y_topleft+1:y_bottomrights+1,:]=pick_colour()
    
    return fresh_canvas

def pick_colour():
    
    red = [1,0,0]
    black=[0,0,0]
    yellow=[1,1,0]
    blue=[0,0,1]
    light_blue=[0,0.75,1]
    orange= [1,0.6,0]
    grey = [0.8,0.8,0.8]
    white = [1,1,1]
    
    colors = np.concatenate((red, black, yellow, blue, light_blue, orange, grey, white), axis=0)
    colors=np.reshape(a=colors, newshape=(8,3))
    # for a slight variation in colour
    # colors = colors + np.random.normal(0,0.001,size=colors.shape)
    np.random.shuffle(colors)
    return colors[0]


### RUN BODY OF SCRIPT ###
if __name__ == "__main__":

    parser = ArgumentParser(description="Produce N mondrian pics in /mondrian_images subdirectory. Default is 100.")
    parser.add_argument("-n", "--number", help="Number N of pics you want. Default is 100", default=100)
    args = parser.parse_args()

    num_iters = int(args.number)
    
    for i in tqdm(range(num_iters), position=0, leave=True):

        #generating canvas
        how_many_lines_v = np.random.randint(5, 9)
        how_many_lines_h = np.random.randint(5, 9)
        num_lines_total = how_many_lines_h+how_many_lines_v
        number_of_corners = (num_lines_total+1)**2

        canvas, width, height = create_canvas()

        canvas, h, v = draw_lines(canvas, width, height, 
            number_of_vertical_lines=how_many_lines_v, number_of_horizontal_lines=how_many_lines_h)

        toplefts, bottomrights = location_corners_better(h, v, width, height)

        #decide which corners we want to paint in
        corners = choose_corners_normal(toplefts, bottomrights)
        funky_corners = choose_corners_funky(toplefts, bottomrights)

        #paint in corners
        fresh_canvas = populate_colours(canvas, corners, how_many_patches=int(num_lines_total*2))
        fresh_canvas = populate_colours(fresh_canvas, funky_corners, how_many_patches=int(num_lines_total/4))

        fig, ax = plt.subplots()
        fig.set_dpi(100)
        ax.imshow(fresh_canvas)
        ax.set_xticks([])
        ax.set_yticks([])

        location = "mondrian_images/"+str(i)+".png"

        plt.savefig(location)
        plt.close()