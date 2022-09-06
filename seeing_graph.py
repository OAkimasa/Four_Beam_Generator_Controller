import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.optimize as opt
import aotools
import poppy
import astropy.units as u

roop_num = 100

N_pix = 128 # number of pixels
nx_size = 1.0 # meter 
pixel_scale = nx_size / float(N_pix) # meter/pix
wl_laser = 589.0 # laser wavelength in nm
t_alt = np.array([0., 500., 1000., 2000., 4000., 8000., 16000.]) # Turbulence height in m
EL = np.array([30.0, 60.0, 90.0]) # telescope elevation in deg
X = (1.0/np.cos(np.radians(90.0-EL))) # Airmass

x,y = np.meshgrid(np.arange(0,N_pix,1), np.arange(0,N_pix,1))
R = np.sqrt((x-N_pix/2.)**2 + (y-N_pix/2.)**2)

w0 = 0.112 # m
w0_pix = w0 * math.sqrt(2.0) / pixel_scale # multiplyt by sqrt(2) to adjust the waist size in the fresnel propagator
gauss =np.exp(-2*R**2/w0_pix**2)


def make_beam_image_10arcsec():
    # mask
    mask_x = np.arange(0, 128, 1)
    mask_y = np.arange(0, 128, 1)
    mask_X, mask_Y = np.meshgrid(mask_x, mask_y)


    r_1 = np.sqrt((mask_X-N_pix/2)**2 + (mask_Y-N_pix/2)**2)
    mask1_radius = 0.247  # m
    mask1 = np.copy(r_1)*0.0
    idx1_1 = np.where(r_1>=mask1_radius/ (1./128.))
    idx1_2 = np.where(r_1<mask1_radius/ (1./128.))
    mask1[idx1_1] = 0.0
    mask1[idx1_2] = 1.0


    r_2 = np.sqrt((mask_X-N_pix/2+0.0685*N_pix)**2 + (mask_Y-N_pix/2+0.0666*N_pix)**2)
    mask2_radius = 0.247  # m
    mask2 = np.copy(r_2)*0.0
    idx2_1 = np.where(r_2>=mask2_radius/ (1./128.))
    idx2_2 = np.where(r_2<mask2_radius/ (1./128.))
    mask2[idx2_1] = 0.0
    mask2[idx2_2] = 1.0


    r_3 = np.sqrt((mask_X-N_pix/2+0.0685*N_pix)**2 + (mask_Y-N_pix/2+0.0666*N_pix)**2)
    mask3_radius = 0.026  # m
    mask3 = np.copy(r_3)*0.0
    idx3_1 = np.where(r_3>=mask3_radius/ (1./128.))
    idx3_2 = np.where(r_3<mask3_radius/ (1./128.))
    mask3[idx3_1] = 1.0
    mask3[idx3_2] = 0.0


    mask4_line = N_pix/2 - 0.138/(1./128.)
    mask4 = np.copy(mask_X)*0.0
    idx4_1 = np.where(mask_X>=mask4_line)
    idx4_2 = np.where(mask_X<mask4_line)
    mask4[idx4_1] = 1.0
    mask4[idx4_2] = 0.0


    mask5_line = N_pix/2 - 0.143/(1./128.)
    mask5 = np.copy(mask_Y)*0.0
    idx5_1 = np.where(mask_Y>=mask5_line)
    idx5_2 = np.where(mask_Y<mask5_line)
    mask5[idx5_1] = 1.0
    mask5[idx5_2] = 0.0

    image = gauss*mask1*mask2*mask3*mask4*mask5
    return image

def make_beam_image_20arcsec():
    # mask
    mask_x = np.arange(0, 128, 1)
    mask_y = np.arange(0, 128, 1)
    mask_X, mask_Y = np.meshgrid(mask_x, mask_y)


    r_1 = np.sqrt((mask_X-N_pix/2)**2 + (mask_Y-N_pix/2)**2)
    mask1_radius = 0.247  # m
    mask1 = np.copy(r_1)*0.0
    idx1_1 = np.where(r_1>=mask1_radius/ (1./128.))
    idx1_2 = np.where(r_1<mask1_radius/ (1./128.))
    mask1[idx1_1] = 0.0
    mask1[idx1_2] = 1.0


    r_2 = np.sqrt((mask_X-N_pix/2+0.0628*N_pix)**2 + (mask_Y-N_pix/2+0.0642*N_pix)**2)
    mask2_radius = 0.247  # m
    mask2 = np.copy(r_2)*0.0
    idx2_1 = np.where(r_2>=mask2_radius/ (1./128.))
    idx2_2 = np.where(r_2<mask2_radius/ (1./128.))
    mask2[idx2_1] = 0.0
    mask2[idx2_2] = 1.0


    r_3 = np.sqrt((mask_X-N_pix/2+0.0628*N_pix)**2 + (mask_Y-N_pix/2+0.0642*N_pix)**2)
    mask3_radius = 0.026  # m
    mask3 = np.copy(r_3)*0.0
    idx3_1 = np.where(r_3>=mask3_radius/ (1./128.))
    idx3_2 = np.where(r_3<mask3_radius/ (1./128.))
    mask3[idx3_1] = 1.0
    mask3[idx3_2] = 0.0


    mask4_line = N_pix/2 - 0.2043/(1./128.)
    mask4 = np.copy(mask_X)*0.0
    idx4_1 = np.where(mask_X>=mask4_line)
    idx4_2 = np.where(mask_X<mask4_line)
    mask4[idx4_1] = 1.0
    mask4[idx4_2] = 0.0


    mask5_line = N_pix/2 - 0.1986/(1./128.)
    mask5 = np.copy(mask_Y)*0.0
    idx5_1 = np.where(mask_Y>=mask5_line)
    idx5_2 = np.where(mask_Y<mask5_line)
    mask5[idx5_1] = 1.0
    mask5[idx5_2] = 0.0

    image = gauss*mask1*mask2*mask3*mask4*mask5
    return image

def make_beam_image_30arcsec():
    # mask
    mask_x = np.arange(0, 128, 1)
    mask_y = np.arange(0, 128, 1)
    mask_X, mask_Y = np.meshgrid(mask_x, mask_y)


    r_1 = np.sqrt((mask_X-N_pix/2)**2 + (mask_Y-N_pix/2)**2)
    mask1_radius = 0.247  # m
    mask1 = np.copy(r_1)*0.0
    idx1_1 = np.where(r_1>=mask1_radius/ (1./128.))
    idx1_2 = np.where(r_1<mask1_radius/ (1./128.))
    mask1[idx1_1] = 0.0
    mask1[idx1_2] = 1.0


    r_2 = np.sqrt((mask_X-N_pix/2-0.0874*N_pix)**2 + (mask_Y-N_pix/2-0.0882*N_pix)**2)
    mask2_radius = 0.253  # m
    mask2 = np.copy(r_2)*0.0
    idx2_1 = np.where(r_2>=mask2_radius/ (1./128.))
    idx2_2 = np.where(r_2<mask2_radius/ (1./128.))
    mask2[idx2_1] = 0.0
    mask2[idx2_2] = 1.0


    r_3 = r_2
    mask3_radius = 0.026  # m
    mask3 = np.copy(r_3)*0.0
    idx3_1 = np.where(r_3>=mask3_radius/ (1./128.))
    idx3_2 = np.where(r_3<mask3_radius/ (1./128.))
    mask3[idx3_1] = 1.0
    mask3[idx3_2] = 0.0


    mask4_line = N_pix/2 - 0.1386/(1./128.)
    mask4 = np.copy(mask_X)*0.0
    idx4_1 = np.where(mask_X>=mask4_line)
    idx4_2 = np.where(mask_X<mask4_line)
    mask4[idx4_1] = 1.0
    mask4[idx4_2] = 0.0


    mask5_line = N_pix/2 - 0.1378/(1./128.)
    mask5 = np.copy(mask_Y)*0.0
    idx5_1 = np.where(mask_Y>=mask5_line)
    idx5_2 = np.where(mask_Y<mask5_line)
    mask5[idx5_1] = 1.0
    mask5[idx5_2] = 0.0

    image = gauss*mask1*mask2*mask3*mask4*mask5
    return image

def make_beam_image_40arcsec():
    # mask
    mask_x = np.arange(0, 128, 1)
    mask_y = np.arange(0, 128, 1)
    mask_X, mask_Y = np.meshgrid(mask_x, mask_y)


    r_1 = np.sqrt((mask_X-N_pix/2)**2 + (mask_Y-N_pix/2)**2)
    mask1_radius = 0.247  # m
    mask1 = np.copy(r_1)*0.0
    idx1_1 = np.where(r_1>=mask1_radius/ (1./128.))
    idx1_2 = np.where(r_1<mask1_radius/ (1./128.))
    mask1[idx1_1] = 0.0
    mask1[idx1_2] = 1.0


    r_2 = np.sqrt((mask_X-N_pix/2-0.1025*N_pix)**2 + (mask_Y-N_pix/2-0.1034*N_pix)**2)
    mask2_radius = 0.253  # m
    mask2 = np.copy(r_2)*0.0
    idx2_1 = np.where(r_2>=mask2_radius/ (1./128.))
    idx2_2 = np.where(r_2<mask2_radius/ (1./128.))
    mask2[idx2_1] = 0.0
    mask2[idx2_2] = 1.0


    r_3 = r_2
    mask3_radius = 0.026  # m
    mask3 = np.copy(r_3)*0.0
    idx3_1 = np.where(r_3>=mask3_radius/ (1./128.))
    idx3_2 = np.where(r_3<mask3_radius/ (1./128.))
    mask3[idx3_1] = 1.0
    mask3[idx3_2] = 0.0


    image = gauss*mask1*mask2*mask3
    return image

def make_phase_screen():
    # total Cn2 [m^1/3] 
    total_cn2 = 5.2770e-13 # m^(1/3), seeing condition: median

    # Wavelength in m
    lam = wl_laser * 1.0e-9 

    # Fried length in m 
    r0_total = (0.423 * (2.0*math.pi/lam)**2 * X * total_cn2)**(-3./5.)
    #print('r0_total = ', r0_total)

    # seeing FWHM in arcsec
    fwhm = np.rad2deg(1.028 * (lam / r0_total))*3600.0 

    # Cn2 fraction in %
    t_frac = np.array([73.16, 6.50, 1.93, 2.52, 5.74, 5.00, 5.15])

    # Cn2 profile in m^(1/3)
    cn2 = total_cn2 * t_frac / 100.0

    # r0 profile in m
    r0 = []
    for  a in X:
        r0.append((0.423 * (2.0*math.pi/lam)**2 * a * cn2)**(-3./5.))
    r0 = np.array(r0)

    # Outer scale in m
    L0 = 30.0 

    phscrn_el = []
    for i in range(len(EL)):
        
        phscrn_arr = []
        for j in range(len(t_alt)):
            phi = aotools.turbulence.infinitephasescreen.PhaseScreenVonKarman(N_pix, pixel_scale, r0[i][j], L0)
            phscrn_arr.append(phi.scrn)
        
        phscrn_el.append(np.array(phscrn_arr))
        
    phscrn_el = np.array(phscrn_el)
    return phscrn_el

def gauss2d(pos, A, x0, y0, w, q, pa):  
    x = pos[0]
    y = pos[1]
    
    xc = (x - x0) * np.cos(np.deg2rad(-pa)) - (y - y0) * np.sin(np.deg2rad(-pa))
    yc = (x - x0) * np.sin(np.deg2rad(-pa)) + (y - y0) * np.cos(np.deg2rad(-pa))
    r = np.sqrt(xc**2 + (yc/q)**2)
    
    g = A * np.exp(-(2.0*r**2/(w**2)))
    
    return g.ravel()

#start_image = gauss
start_image = make_beam_image_40arcsec()
phscrn_el_arr = []
for i in range(roop_num):
    print('Phase screen i = ', i)
    phscrn_el = make_phase_screen()
    phscrn_el_arr.append(phscrn_el)

plt.rcParams['font.size'] = 20
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.size'] = 10
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['ytick.major.width'] = 1.5


# =========================================================================================
w_05km_EL30 = []
w_1km_EL30 = []
w_2km_EL30 = []
w_4km_EL30 = []
w_8km_EL30 = []
w_16km_EL30 = []
w_90km_EL30 = []

w_05km_EL60 = []
w_1km_EL60 = []
w_2km_EL60 = []
w_4km_EL60 = []
w_8km_EL60 = []
w_16km_EL60 = []
w_90km_EL60 = []

w_05km_EL90 = []
w_1km_EL90 = []
w_2km_EL90 = []
w_4km_EL90 = []
w_8km_EL90 = []
w_16km_EL90 = []
w_90km_EL90 = []

count = 0
for j in phscrn_el_arr:
    count += 1
    print('phase screen count = ', count)
    phscrn_el = j # phase maps in radian (it should be same pix scale as the gauss image)

    for i_EL in range(3):
        if i_EL == 0:
            # i: EL, j: t_alt
            # (e.g. set the 0m phase screen for EL=60m => i = 1, j = 0)
            opd = phscrn_el[i_EL][0] * (wl_laser*1.0e-9/(2.0*math.pi)) # Optical path difference in m
            input_wave = poppy.ArrayOpticalElement(transmission=start_image, opd=opd*u.m, name='input wavefront', pixelscale=pixel_scale*u.m/u.pixel)
            #input_wave.display(what='both')

            # Setup initial condition for the fresnel propagation
            wf = poppy.FresnelWavefront(0.5*u.m, wavelength=wl_laser*1.0e-9, npix=N_pix, oversample=4)
            wf *= input_wave

            # 2D gaussian fit
            Xp,Yp = np.meshgrid(np.arange(0,N_pix*4,1), np.arange(0,N_pix*4,1))
            initial_guess = (1.0, 2.0*N_pix, 2.0*N_pix, w0/pixel_scale, 1.0, 0)
            popt, pcov = opt.curve_fit(gauss2d, (Xp,Yp), wf.intensity.ravel(), p0 = initial_guess)
            gauss_fit = gauss2d((Xp,Yp), *popt)
            A_fit,x0_fit,y0_fit,w_fit,q_fit,pa_fit = popt


            w_arr = []
            L_arr = []
            w_arr.append(w_fit*pixel_scale)
            L_arr.append(0.0)


            for i in range(len(t_alt)-1):
                
                # propagation distance 
                dL = (t_alt[i+1] - t_alt[i]) * X[i_EL]  # distance to the next turbulence plane in m
                L_arr.append(t_alt[i+1]*X[i_EL])

                # propagetion
                wf.propagate_fresnel(dL*u.m)

                # 2D gaussian fit
                Xp,Yp = np.meshgrid(np.arange(0,N_pix*4,1), np.arange(0,N_pix*4,1))
                initial_guess = (1.0, 2.0*N_pix, 2.0*N_pix, w0/pixel_scale, 1.0, 0)
                popt, pcov = opt.curve_fit(gauss2d, (Xp,Yp), wf.intensity.ravel(), p0 = initial_guess)
                gauss_fit = gauss2d((Xp,Yp), *popt)
                A_fit,x0_fit,y0_fit,w_fit,q_fit,pa_fit = popt
                w_arr.append(w_fit*pixel_scale)
                print("Waist size = %.4f" % (w_fit*pixel_scale), t_alt[i+1], "i=", i)
                if i == 0:
                    w_05km_EL30.append(w_fit*pixel_scale)
                elif i == 1:
                    w_1km_EL30.append(w_fit*pixel_scale)
                elif i == 2:
                    w_2km_EL30.append(w_fit*pixel_scale)
                elif i == 3:
                    w_4km_EL30.append(w_fit*pixel_scale)
                elif i == 4:
                    w_8km_EL30.append(w_fit*pixel_scale)
                elif i == 5:
                    w_16km_EL30.append(w_fit*pixel_scale)

                # add phase delay
                ph = poppy.ArrayOpticalElement(transmission=(start_image*0.0+1.0), opd=phscrn_el[i_EL][i+1] * (wl_laser*1.0e-9/(2.0*math.pi)), name='', pixelscale=pixel_scale*u.m/u.pixel)
                #ph.display(what='both')
                wf *= ph

            # last propagation until the height of the Na layer (H=90km) 
            dL = (90.0e+3 - t_alt[len(t_alt)-1]) * X[i_EL]

            # propagetion
            wf.propagate_fresnel(dL*u.m)

            # 2D gaussian fit
            Xp,Yp = np.meshgrid(np.arange(0,N_pix*4,1), np.arange(0,N_pix*4,1))
            initial_guess = (1.0, 2.0*N_pix, 2.0*N_pix, w0/pixel_scale, 1.0, 0)
            popt, pcov = opt.curve_fit(gauss2d, (Xp,Yp), wf.intensity.ravel(), p0 = initial_guess, maxfev=10000)
            gauss_fit = gauss2d((Xp,Yp), *popt)
            A_fit,x0_fit,y0_fit,w_fit,q_fit,pa_fit = popt
            print("Waist size = %.4f" % (w_fit*pixel_scale), "90km")
            w_90km_EL30.append(w_fit*pixel_scale)
        elif i_EL == 1:
            # i: EL, j: t_alt
            # (e.g. set the 0m phase screen for EL=60m => i = 1, j = 0)
            opd = phscrn_el[i_EL][0] * (wl_laser*1.0e-9/(2.0*math.pi)) # Optical path difference in m
            input_wave = poppy.ArrayOpticalElement(transmission=start_image, opd=opd*u.m, name='input wavefront', pixelscale=pixel_scale*u.m/u.pixel)
            #input_wave.display(what='both')

            # Setup initial condition for the fresnel propagation
            wf = poppy.FresnelWavefront(0.5*u.m, wavelength=wl_laser*1.0e-9, npix=N_pix, oversample=4)
            wf *= input_wave

            # 2D gaussian fit
            Xp,Yp = np.meshgrid(np.arange(0,N_pix*4,1), np.arange(0,N_pix*4,1))
            initial_guess = (1.0, 2.0*N_pix, 2.0*N_pix, w0/pixel_scale, 1.0, 0)
            popt, pcov = opt.curve_fit(gauss2d, (Xp,Yp), wf.intensity.ravel(), p0 = initial_guess)
            gauss_fit = gauss2d((Xp,Yp), *popt)
            A_fit,x0_fit,y0_fit,w_fit,q_fit,pa_fit = popt


            w_arr = []
            L_arr = []
            w_arr.append(w_fit*pixel_scale)
            L_arr.append(0.0)


            for i in range(len(t_alt)-1):
                
                # propagation distance 
                dL = (t_alt[i+1] - t_alt[i]) * X[i_EL]  # distance to the next turbulence plane in m
                L_arr.append(t_alt[i+1]*X[i_EL])

                # propagetion
                wf.propagate_fresnel(dL*u.m)

                # 2D gaussian fit
                Xp,Yp = np.meshgrid(np.arange(0,N_pix*4,1), np.arange(0,N_pix*4,1))
                initial_guess = (1.0, 2.0*N_pix, 2.0*N_pix, w0/pixel_scale, 1.0, 0)
                popt, pcov = opt.curve_fit(gauss2d, (Xp,Yp), wf.intensity.ravel(), p0 = initial_guess)
                gauss_fit = gauss2d((Xp,Yp), *popt)
                A_fit,x0_fit,y0_fit,w_fit,q_fit,pa_fit = popt
                w_arr.append(w_fit*pixel_scale)
                print("Waist size = %.4f" % (w_fit*pixel_scale), t_alt[i+1], "i=", i)
                if i == 0:
                    w_05km_EL60.append(w_fit*pixel_scale)
                elif i == 1:
                    w_1km_EL60.append(w_fit*pixel_scale)
                elif i == 2:
                    w_2km_EL60.append(w_fit*pixel_scale)
                elif i == 3:
                    w_4km_EL60.append(w_fit*pixel_scale)
                elif i == 4:
                    w_8km_EL60.append(w_fit*pixel_scale)
                elif i == 5:
                    w_16km_EL60.append(w_fit*pixel_scale)

                # add phase delay
                ph = poppy.ArrayOpticalElement(transmission=(start_image*0.0+1.0), opd=phscrn_el[i_EL][i+1] * (wl_laser*1.0e-9/(2.0*math.pi)), name='', pixelscale=pixel_scale*u.m/u.pixel)
                #ph.display(what='both')
                wf *= ph

            # last propagation until the height of the Na layer (H=90km) 
            dL = (90.0e+3 - t_alt[len(t_alt)-1]) * X[i_EL]

            # propagetion
            wf.propagate_fresnel(dL*u.m)

            # 2D gaussian fit
            Xp,Yp = np.meshgrid(np.arange(0,N_pix*4,1), np.arange(0,N_pix*4,1))
            initial_guess = (1.0, 2.0*N_pix, 2.0*N_pix, w0/pixel_scale, 1.0, 0)
            popt, pcov = opt.curve_fit(gauss2d, (Xp,Yp), wf.intensity.ravel(), p0 = initial_guess, maxfev=10000)
            gauss_fit = gauss2d((Xp,Yp), *popt)
            A_fit,x0_fit,y0_fit,w_fit,q_fit,pa_fit = popt
            print("Waist size = %.4f" % (w_fit*pixel_scale), "90km")
            w_90km_EL60.append(w_fit*pixel_scale)
        elif i_EL == 2:
            # i: EL, j: t_alt
            # (e.g. set the 0m phase screen for EL=60m => i = 1, j = 0)
            opd = phscrn_el[i_EL][0] * (wl_laser*1.0e-9/(2.0*math.pi)) # Optical path difference in m
            input_wave = poppy.ArrayOpticalElement(transmission=start_image, opd=opd*u.m, name='input wavefront', pixelscale=pixel_scale*u.m/u.pixel)
            #input_wave.display(what='both')

            # Setup initial condition for the fresnel propagation
            wf = poppy.FresnelWavefront(0.5*u.m, wavelength=wl_laser*1.0e-9, npix=N_pix, oversample=4)
            wf *= input_wave

            # 2D gaussian fit
            Xp,Yp = np.meshgrid(np.arange(0,N_pix*4,1), np.arange(0,N_pix*4,1))
            initial_guess = (1.0, 2.0*N_pix, 2.0*N_pix, w0/pixel_scale, 1.0, 0)
            popt, pcov = opt.curve_fit(gauss2d, (Xp,Yp), wf.intensity.ravel(), p0 = initial_guess)
            gauss_fit = gauss2d((Xp,Yp), *popt)
            A_fit,x0_fit,y0_fit,w_fit,q_fit,pa_fit = popt


            w_arr = []
            L_arr = []
            w_arr.append(w_fit*pixel_scale)
            L_arr.append(0.0)


            for i in range(len(t_alt)-1):
                
                # propagation distance 
                dL = (t_alt[i+1] - t_alt[i]) * X[i_EL]  # distance to the next turbulence plane in m
                L_arr.append(t_alt[i+1]*X[i_EL])

                # propagetion
                wf.propagate_fresnel(dL*u.m)

                # 2D gaussian fit
                Xp,Yp = np.meshgrid(np.arange(0,N_pix*4,1), np.arange(0,N_pix*4,1))
                initial_guess = (1.0, 2.0*N_pix, 2.0*N_pix, w0/pixel_scale, 1.0, 0)
                popt, pcov = opt.curve_fit(gauss2d, (Xp,Yp), wf.intensity.ravel(), p0 = initial_guess)
                gauss_fit = gauss2d((Xp,Yp), *popt)
                A_fit,x0_fit,y0_fit,w_fit,q_fit,pa_fit = popt
                w_arr.append(w_fit*pixel_scale)
                print("Waist size = %.4f" % (w_fit*pixel_scale), t_alt[i+1], "i=", i)
                if i == 0:
                    w_05km_EL90.append(w_fit*pixel_scale)
                elif i == 1:
                    w_1km_EL90.append(w_fit*pixel_scale)
                elif i == 2:
                    w_2km_EL90.append(w_fit*pixel_scale)
                elif i == 3:
                    w_4km_EL90.append(w_fit*pixel_scale)
                elif i == 4:
                    w_8km_EL90.append(w_fit*pixel_scale)
                elif i == 5:
                    w_16km_EL90.append(w_fit*pixel_scale)

                # add phase delay
                ph = poppy.ArrayOpticalElement(transmission=(start_image*0.0+1.0), opd=phscrn_el[i_EL][i+1] * (wl_laser*1.0e-9/(2.0*math.pi)), name='', pixelscale=pixel_scale*u.m/u.pixel)
                #ph.display(what='both')
                wf *= ph

            # last propagation until the height of the Na layer (H=90km) 
            dL = (90.0e+3 - t_alt[len(t_alt)-1]) * X[i_EL]

            # propagetion
            wf.propagate_fresnel(dL*u.m)

            # 2D gaussian fit
            Xp,Yp = np.meshgrid(np.arange(0,N_pix*4,1), np.arange(0,N_pix*4,1))
            initial_guess = (1.0, 2.0*N_pix, 2.0*N_pix, w0/pixel_scale, 1.0, 0)
            popt, pcov = opt.curve_fit(gauss2d, (Xp,Yp), wf.intensity.ravel(), p0 = initial_guess, maxfev=10000)
            gauss_fit = gauss2d((Xp,Yp), *popt)
            A_fit,x0_fit,y0_fit,w_fit,q_fit,pa_fit = popt
            print("Waist size = %.4f" % (w_fit*pixel_scale), "90km")
            w_90km_EL90.append(w_fit*pixel_scale)


# 2D gaussian fit (average)
w_05km_EL30 = [i for i in w_05km_EL30 if i >= 0.10]
w_1km_EL30 = [i for i in w_1km_EL30 if i >= 0.10]
w_2km_EL30 = [i for i in w_2km_EL30 if i >= 0.10]
w_4km_EL30 = [i for i in w_4km_EL30 if i >= 0.10]
w_8km_EL30 = [i for i in w_8km_EL30 if i >= 0.10]
w_16km_EL30 = [i for i in w_16km_EL30 if i >= 0.10]
w_90km_EL30 = [i for i in w_90km_EL30 if i >= 0.10]
print(w_05km_EL30, "w_05km_EL30")
print(w_1km_EL30, "w_1km_EL30")
print(w_2km_EL30, "w_2km_EL30")
print(w_4km_EL30, "w_4km_EL30")
print(w_8km_EL30, "w_8km_EL30")
print(w_16km_EL30, "w_16km_EL30")
print(w_90km_EL30, "w_90km_EL30")
w_arr_average_EL30 = [np.mean(np.abs(w_05km_EL30)), np.mean(np.abs(w_1km_EL30)), np.mean(np.abs(w_2km_EL30)), np.mean(np.abs(w_4km_EL30)), np.mean(np.abs(w_8km_EL30)), np.mean(np.abs(w_16km_EL30)), np.mean(np.abs(w_90km_EL30))]
print(w_arr_average_EL30, "w_arr_average_EL30")
error_arr_average_EL30 = [np.std(w_05km_EL30)/len(w_05km_EL30), np.std(w_1km_EL30)/len(w_1km_EL30), np.std(w_2km_EL30)/len(w_2km_EL30), np.std(w_4km_EL30)/len(w_4km_EL30), np.std(w_8km_EL30)/len(w_8km_EL30), np.std(w_16km_EL30)/len(w_16km_EL30), np.std(w_90km_EL30)/len(w_90km_EL30)]

w_05km_EL60 = [i for i in w_05km_EL60 if i >= 0.10]
w_1km_EL60 = [i for i in w_1km_EL60 if i >= 0.10]
w_2km_EL60 = [i for i in w_2km_EL60 if i >= 0.10]
w_4km_EL60 = [i for i in w_4km_EL60 if i >= 0.10]
w_8km_EL60 = [i for i in w_8km_EL60 if i >= 0.10]
w_16km_EL60 = [i for i in w_16km_EL60 if i >= 0.10]
w_90km_EL60 = [i for i in w_90km_EL60 if i >= 0.10]
print(w_05km_EL60, "w_05km_EL60")
print(w_1km_EL60, "w_1km_EL60")
print(w_2km_EL60, "w_2km_EL60")
print(w_4km_EL60, "w_4km_EL60")
print(w_8km_EL60, "w_8km_EL60")
print(w_16km_EL60, "w_16km_EL60")
print(w_90km_EL60, "w_90km_EL60")
w_arr_average_EL60 = [np.mean(np.abs(w_05km_EL60)), np.mean(np.abs(w_1km_EL60)), np.mean(np.abs(w_2km_EL60)), np.mean(np.abs(w_4km_EL60)), np.mean(np.abs(w_8km_EL60)), np.mean(np.abs(w_16km_EL60)), np.mean(np.abs(w_90km_EL60))]
print(w_arr_average_EL60, "w_arr_average_EL60")
error_arr_average_EL60 = [np.std(w_05km_EL60)/len(w_05km_EL60), np.std(w_1km_EL60)/len(w_1km_EL60), np.std(w_2km_EL60)/len(w_2km_EL60), np.std(w_4km_EL60)/len(w_4km_EL60), np.std(w_8km_EL60)/len(w_8km_EL60), np.std(w_16km_EL60)/len(w_16km_EL60), np.std(w_90km_EL60)/len(w_90km_EL60)]

w_05km_EL90 = [i for i in w_05km_EL90 if i >= 0.10]
w_1km_EL90 = [i for i in w_1km_EL90 if i >= 0.10]
w_2km_EL90 = [i for i in w_2km_EL90 if i >= 0.10]
w_4km_EL90 = [i for i in w_4km_EL90 if i >= 0.10]
w_8km_EL90 = [i for i in w_8km_EL90 if i >= 0.10]
w_16km_EL90 = [i for i in w_16km_EL90 if i >= 0.10]
w_90km_EL90 = [i for i in w_90km_EL90 if i >= 0.10]
print(w_05km_EL90, "w_05km_EL90")
print(w_1km_EL90, "w_1km_EL90")
print(w_2km_EL90, "w_2km_EL90")
print(w_4km_EL90, "w_4km_EL90")
print(w_8km_EL90, "w_8km_EL90")
print(w_16km_EL90, "w_16km_EL90")
print(w_90km_EL90, "w_90km_EL90")
w_arr_average_EL90 = [np.mean(np.abs(w_05km_EL90)), np.mean(np.abs(w_1km_EL90)), np.mean(np.abs(w_2km_EL90)), np.mean(np.abs(w_4km_EL90)), np.mean(np.abs(w_8km_EL90)), np.mean(np.abs(w_16km_EL90)), np.mean(np.abs(w_90km_EL90))]
print(w_arr_average_EL90, "w_arr_average_EL90")
error_arr_average_EL90 = [np.std(w_05km_EL90)/len(w_05km_EL90), np.std(w_1km_EL90)/len(w_1km_EL90), np.std(w_2km_EL90)/len(w_2km_EL90), np.std(w_4km_EL90)/len(w_4km_EL90), np.std(w_8km_EL90)/len(w_8km_EL90), np.std(w_16km_EL90)/len(w_16km_EL90), np.std(w_90km_EL90)/len(w_90km_EL90)]


"""plt.figure(figsize=(10,10))
#plt.plot(L_arr, w_arr,"*", label='w/o Turbulence')
plt.errorbar(L_arr_average, w_arr_average, yerr=error_arr_average, fmt="*", capsize=3, label='w/o Turbulence')"""
#plt.plot(L_arr, w_atm_arr,"*", label='with Turbulence')
"""z = np.arange(0,125.0e+3, 100)
wz = w0 * np.sqrt(1.0+(wl_laser*1.0e-9*z/(math.pi*w0**2))**2)
plt.plot(z,wz)"""
# ==================================================================
total_cn2 = 5.277e-13 # m^(1/3), 50% median seeing condition?

# Wavelength in m
lam = 589.0 * 1.0e-9 

# telescope elevation in deg
EL = np.array([30.0, 60.0, 90.0])

# Airmass 
X = (1.0/np.cos(np.radians(90.0-EL)))

# Fried length in m 
r0_total = (0.423 * (2.0*math.pi/lam)**2 * X * total_cn2)**(-3./5.)

# seeing FWHM in arcsec
fwhm = np.rad2deg(1.028 * (lam / r0_total))*3600.0 

# Turbulence height in m
t_alt  = np.array([0., 500., 1000., 2000., 4000., 8000., 16000.])

# Cn2 fraction in %
t_frac = np.array([73.16, 6.50, 1.93, 2.52, 5.47, 5.00, 5.15])

# Cn2 profile in m^(1/3)
cn2 = total_cn2 * t_frac / 100.0

# r0 profile in m
r0 = []
for  a in X:
    r0.append((0.423 * (2.0*math.pi/lam)**2 * a * cn2)**(-3./5.))
r0 = np.array(r0)

z = np.arange(0,200.0e+3, 100)

w0 = 0.112 # initial beam waist radius in m
D = 0.3    # Laser gaussian beam diameter in m
H_Na = 90.0e+3 # Sodium Layer height in m


fig = plt.figure(figsize=(11,8))



ax = fig.add_subplot(1,1,1)

#pid = 131 + i 
#ax.append(fig.add_subplot(pid))
ax.set_xlabel('Propagation distance [km]',fontsize=15)
ax.set_ylabel('Beam waist radius [m]',fontsize=15)
ax.set_title('%.0f roop' % roop_num,fontsize=15)
ax.tick_params(axis='both', labelsize=15)
#ax.set_xlim(0,1.11112*H_Na/math.cos(np.radians(90.0-EL[i]))/1000.0)
ax.set_xlim(0,200)
ax.grid()

w0 = 0.112 # initial beam waist radius 
D = 0.3    # Laser gaussian beam diameter 

L_arr_average_EL30 = np.array([0.5, 1., 2., 4., 8., 16., 90.])*X[0]  # km
print(L_arr_average_EL30, "L_arr_average_EL30")
ax.errorbar(L_arr_average_EL30, w_arr_average_EL30, yerr=error_arr_average_EL30, fmt="*", capsize=3, color="blue", ecolor="blue")

L_arr_average_EL60 = np.array([0.5, 1., 2., 4., 8., 16., 90.])*X[1]  # km
print(L_arr_average_EL60, "L_arr_average_EL60")
ax.errorbar(L_arr_average_EL60, w_arr_average_EL60, yerr=error_arr_average_EL60, fmt="*", capsize=3, color="green", ecolor="green")

L_arr_average_EL90 = np.array([0.5, 1., 2., 4., 8., 16., 90.])*X[2]  # km
print(L_arr_average_EL90, "L_arr_average_EL90")
ax.errorbar(L_arr_average_EL90, w_arr_average_EL90, yerr=error_arr_average_EL90, fmt="*", capsize=3, color="red", ecolor="red")



z = np.arange(0,200.0e+3, 100)
wz = w0 * np.sqrt(1.0+(lam*z/(math.pi*w0**2))**2) # Gaussian beam theoretical diffraction
ax.plot(z/1000.0,wz,'--', c='black', label='Diffraction')  # 'Diffraction'

for i in range(3):
    fwhm_seeing = 1.082 * (lam/r0_total[i]) * (180.0/math.pi) * 3600.0 # Seeing FWHM in arcsec
    ws = (fwhm_seeing / math.sqrt(2.0*math.log(2.0)))*(math.pi/180.0)*(1.0/3600.0)*z
    wsz = np.sqrt(wz**2 + ws**2)

    """s2_jitter = 0.182 * lam**2 * D**(-1.0/3.0) * r0_total[i]**(-5.0/3.0) # rad in RMS
    fwhm_jitter = math.sqrt(s2_jitter) * 2.355 * (180.0/math.pi) * 3600.0 
    fwhm_mod = math.sqrt(fwhm_seeing**2 - fwhm_jitter**2)
    wsr = (fwhm_mod / math.sqrt(2.0*math.log(2.0)))*(math.pi/180.0)*(1.0/3600.0)*z
    wsrz = np.sqrt(wz**2 + wsr**2)
    ax.plot(z/1000.0,wsrz,'--',c='blue', label='Diff.+Seeing(Kolmogorov jitter removed)')  # 'Diff.+Seeing(Kolmogorov jitter removed)'"""

    L = H_Na * X[i]  

    rc2 = (2.87)*w0**(-1.0/3.0)*L**2*np.sum(cn2*X[i]*(1.0-t_alt*X[i]/L)**2)
    fwhm_rc = np.sqrt((2.87)*w0**(-1.0/3.0)*np.sum(cn2*X[i]*(1.0-t_alt*X[i]/L)**2)) * 2.355 * (180.0/math.pi) * 3600.0
    fwhm_mod2 = math.sqrt(fwhm_seeing**2 - (fwhm_rc)**2)

    wsr2 = (fwhm_mod2 / math.sqrt(2.0*math.log(2.0)))*(math.pi/180.0)*(1.0/3600.0)*z
    wsrz2 = np.sqrt(wz**2 + wsr2**2)
    if i == 0:
        ax.plot(z/1000.0,wsz,'-',c='blue')  # 'Diff.+Seeing(empirical)'
        ax.plot(z/1000.0,wsrz2,'-.',c='blue', label='Diff+Seeing(Gaussian jitter removed) (EL30)')  # 'Diff+Seeing(Gaussian jitter removed)'
    elif i == 1:
        ax.plot(z/1000.0,wsz,'-',c='green')
        ax.plot(z/1000.0,wsrz2,'-.',c='green', label='Diff+Seeing(Gaussian jitter removed) (EL60)')
    elif i == 2:
        ax.plot(z/1000.0,wsz,'-',c='red')
        ax.plot(z/1000.0,wsrz2,'-.',c='red', label='Diff+Seeing(Gaussian jitter removed) (EL90)')


#print(EL[i], fwhm_seeing, fwhm_jitter, fwhm_rc, rc2)
# ==================================================================

#plt.ylabel('Beam waist radius [m]')
#plt.xlabel('Propagation Distance [m]')
plt.legend()
plt.grid()


plt.show()