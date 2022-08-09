import numpy as np
import math
import pandas as pd
from raytrace_lib import raytrace

### Constants ### 
t_bs = 6.0      # Beam Splitter substrate thickness [mm]
n_bs = 1.458    # Refractive index of the BS substrate (Fused Silica at 589nm)
n_prism = 1.458 # Refractive index of the Prism (Fused Silica at 589nm)
n_air = 1.0     # Refractive index of the Air

### global variables for the ray equation solver ### 
ray_num_to_calc = 1
ray_target = [0, 0, 0, 0]

### Read 4beam mirror position/angle table ###
posang_csv = pd.read_csv(filepath_or_buffer="4beam_mirror_pos_angle_default.csv", sep=",", index_col=0)

### Beam reflected by the FM4-1 (Initial incident ray parameters) ###
r_ini_dir = np.array([1.0, 0.0, 0.0])
r_ini_pos = np.array([posang_csv.at['FM4-1','X (mm)'],posang_csv.at['FM4-1','Y (mm)'], posang_csv.at['FM4-1','Z (mm)']])

### Ray trace at the Beam Splitters ###
def bs_raytrace():

    # Get global variables 
    global r_ini_dir
    global r_ini_Pos
    global pos_ang_csv
    global t_bs
    global n_bs
    global n_air

    # define BS surfaces 
    bs_surface = ['BS1', 'BS2', 'BS3', 'BS4']

    # initialize the input ray
    r_in_dir = r_ini_dir
    r_in_pos = r_ini_pos

    # initialize the output ray arrays 
    r_bs_dir = []  # BS refelcted ray unit vector 
    r_bs_pos = []  # BS reflected ray origin (position)

    # Compute the ray at the BS
    for i in range(len(bs_surface)):

        # TX/TY column name
        tx_col_name = 'TX (deg), 10asec'
        ty_col_name = 'TY (deg), 10asec'

        # surface parameters
        tx = posang_csv.at[bs_surface[i], 'TX (deg)']
        ty = posang_csv.at[bs_surface[i], 'TY (deg)']
        px = posang_csv.at[bs_surface[i], 'X (mm)']
        py = posang_csv.at[bs_surface[i], 'Y (mm)']
        pz = posang_csv.at[bs_surface[i], 'Z (mm)']
        txy = np.array([np.deg2rad(tx), np.deg2rad(ty)])
        pos = np.array([px, py, pz])
        
        # Compute reflection vector and origin of the ray at the BS front surface 
        rt = raytrace()
        rt.input_ray_dir = r_in_dir
        rt.input_ray_pos = r_in_pos
        rt.surface_txy = txy
        rt.surface_pos = pos
        rt.reflect()

        # Save the reflected ray parameters  
        r_bs_dir.append(rt.output_ray_dir)
        r_bs_pos.append(rt.output_ray_pos)

        if i < len(bs_surface):
            # Compute refracted ray at the BS front surface 
            rt.n1 = n_air
            rt.n2 = n_bs
            rt.refract()

            # BS back surface parameters
            norm = rt.calc_norm(txy)
            back_txy = txy
            back_pos = pos - t_bs * norm

            # Compute refraction vector and origin of the ray at the BS back surface 
            rt_bs = raytrace()
            rt_bs.n1 = n_bs
            rt_bs.n2 = n_air
            rt_bs.input_ray_dir = rt.output_ray_dir
            rt_bs.input_ray_pos = rt.output_ray_pos
            rt_bs.surface_txy = back_txy
            rt_bs.surface_pos = back_pos
            rt_bs.refract()

            # Update the input ray
            r_in_dir = rt_bs.output_ray_dir
            r_in_pos = rt_bs.output_ray_pos

    return r_bs_dir, r_bs_pos # return dir/pos of the BS reflected rays

### Ray trace for each of the 4-beam rays ###
def fbr_raytrace(ray_num, tx_ttm1, ty_ttm1, tx_ttm2, ty_ttm2):

    # Get global variables 
    global pos_ang_csv
    global n_prism
    global n_air

    # define optical surfaces
    s1 = 'TTM1-%d' % (ray_num)
    s2 = 'TTM2-%d' % (ray_num)
    s3 = 'PRISM-%d_Front' % (ray_num)
    s4 = 'PRISM_Back'
    s5 = 'FBM'

    surface = [s1, s2, s3, s4, s5]
    surface_flag = [0, 0, 1, 1, 0] # 0: reflection, 1: refraction

    # calculate input rays
    r_bs_dir, r_bs_pos = bs_raytrace()
    
    # input rays 
    r_in_dir = r_bs_dir[ray_num-1]
    r_in_pos = r_bs_pos[ray_num-1]
    print(r_in_dir)
    print(r_in_pos)
    
    ray_param_surface = []
    for i in range(len(surface)):

        # surface parameters
        if surface[i].find('TTM1') >= 0:
            tx = tx_ttm1
            ty = ty_ttm1
        elif surface[i].find('TTM2') >= 0:
            tx = tx_ttm2
            ty = ty_ttm2            
        px = posang_csv.at[surface[i], 'X (mm)']
        py = posang_csv.at[surface[i], 'Y (mm)']
        pz = posang_csv.at[surface[i], 'Z (mm)']
        txy = np.array([np.deg2rad(tx), np.deg2rad(ty)])
        pos = np.array([px, py, pz])
    
        # Compute reflection/refraction vector and origin of the ray at the optical surface
        rt = raytrace()
        rt.input_ray_dir = r_in_dir
        rt.input_ray_pos = r_in_pos
        rt.surface_txy = txy
        rt.surface_pos = pos
        if surface_flag[i] == 0:
            # calculate reflection vector
            rt.reflect()   
        else:
            # Set refraction index
            if surface[i].find('Front') >= 0:
                rt.n1 = n_air
                rt.n2 = n_prism
            elif surface[i].find('Back') >= 0:
                rt.n1 = n_prism
                rt.n2 = n_air
            # calculate refraction vector 
            rt.refract()   

        # Calculate the distance between the ray origin and the surface center 
        r_d = math.sqrt((rt.output_ray_pos[0] - pos[0])**2 + (rt.output_ray_pos[1] - pos[1])**2 + (rt.output_ray_pos[2] - pos[2])**2)
            
        # Save the ray parameters
        rt_res = np.array([surface[i], rt.output_ray_dir, rt.output_ray_pos, r_d])
        ray_param_surface.append(rt_res)

        # Update the input ray parameters 
        r_in_dir = rt.output_ray_dir
        r_in_pos = rt.output_ray_pos


    # Calculate the ray position at the Beam Expander (BE) entrance pupil
    px = posang_csv.at['BE_entrance_pupil', 'X (mm)']
    py = posang_csv.at['BE_entrance_pupil', 'Y (mm)']
    pz = posang_csv.at['BE_entrance_pupil', 'Z (mm)']

    t = (px - r_in_pos[0]) / r_in_dir[0]
    x_pupil = px
    y_pupil = r_in_pos[1] + t * r_in_dir[1]
    z_pupil = r_in_pos[2] + t * r_in_dir[2]
    r_d = math.sqrt((x_pupil - px)**2 + (y_pupil - py)**2 + (z_pupil - pz)**2)
        
    # Save the final ray parameters at the BE entrance pupil 
    rt_res = np.array(['BE_entrance_pupil', r_in_dir,  np.array([x_pupil, y_pupil, z_pupil]), r_d])
    ray_param_surface.append(rt_res)

    return ray_param_surface

### Ray equations to solve ### 
def rayequation(x):

    global ray_num_to_calc
    global target

    tx_ttm1 = x[0]
    ty_ttm1 = x[1]
    tx_ttm2 = x[2]
    ty_ttm2 = x[3]
    
    ray_param = fbr_raytrace(ray_num_to_calc, tx_ttm1, ty_ttm1, tx_ttm2, ty_ttm2)
    print(ray_param)



def main():
    global ray_num_to_calc
    global target
    
    rayequation([-135.047700, -3.74360, 137.5387, -3.495310])
    
    
if __name__ == '__main__':
  main()