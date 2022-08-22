import numpy as np

def generate_TTM_nV(arcsec_nV_to_ray):
    #arcsec_nV_to_ray = arcsec_ray_to_ray/2
    TTM1_ray1 = [-0.704396 - 0.0000108186*arcsec_nV_to_ray + 8.6129*(10**-7)*arcsec_nV_to_ray**2,
                -0.706898 + 0.0000674041*arcsec_nV_to_ray - 0.00000353366*arcsec_nV_to_ray**2,
                -0.0642305 - 0.00060405*arcsec_nV_to_ray + 0.0000285568*arcsec_nV_to_ray**2]
    TTM1_ray2 = [-0.711624 + 0.0000629776*arcsec_nV_to_ray - 0.00000298822*arcsec_nV_to_ray**2,
                - 0.699791 - 0.0000527309*arcsec_nV_to_ray + 0.00000244596*arcsec_nV_to_ray**2,
                - 0.0623188 - 0.000126143*arcsec_nV_to_ray + 0.00000661246*arcsec_nV_to_ray**2]
    TTM1_ray3 = [-0.704355 - 0.0000739697*arcsec_nV_to_ray + 0.00000347315*arcsec_nV_to_ray**2,
                -0.706878 + 0.0000309578*arcsec_nV_to_ray - 0.00000137244*arcsec_nV_to_ray**2,
                0.0648759 - 0.000476259*arcsec_nV_to_ray + 0.000023256*arcsec_nV_to_ray**2]
    TTM1_ray4 =[-0.709897 + 0.0000283594*arcsec_nV_to_ray - 0.00000234672*arcsec_nV_to_ray**2,
                -0.701653 - 0.0000291465*arcsec_nV_to_ray + 0.00000248545*arcsec_nV_to_ray**2,
                0.0610613 - 0.00000464994*arcsec_nV_to_ray + 0.00000124638*arcsec_nV_to_ray**2]
    TTM2_ray1 = [0.674683 - 0.0000456866*arcsec_nV_to_ray - 2.618*(10**-7)*arcsec_nV_to_ray**2,
                -0.735565 - 0.0000797398*arcsec_nV_to_ray + 0.00000198311*arcsec_nV_to_ray**2,
                -0.0611784 + 0.00045241*arcsec_nV_to_ray - 0.0000266337*arcsec_nV_to_ray**2]
    TTM2_ray2 = [0.758222 + 0.0000783648*arcsec_nV_to_ray - 0.00000*arcsec_nV_to_ray**2,
                -0.649361 + 0.000096067*arcsec_nV_to_ray - 9.5498*(10**-7)*arcsec_nV_to_ray**2,
                -0.0585671 - 0.0000510687*arcsec_nV_to_ray - 0.00000237242*arcsec_nV_to_ray**2]
    TTM2_ray3 = [0.674731 - 0.0000921969*arcsec_nV_to_ray + 0.00000153376*arcsec_nV_to_ray**2,
                -0.735585 - 0.0000247856*arcsec_nV_to_ray - 0.00000102432*arcsec_nV_to_ray**2,
                0.060531 + 0.000688423*arcsec_nV_to_ray - 0.0000280453*arcsec_nV_to_ray**2]
    TTM2_ray4 = [0.759895 + 0.0000313842*arcsec_nV_to_ray + 1.374*(10**-7)*arcsec_nV_to_ray**2,
                -0.64731 + 0.0000557544*arcsec_nV_to_ray - 4.0025*(10**-7)*arcsec_nV_to_ray**2,
                0.0595894 + 0.000201299*arcsec_nV_to_ray - 0.00000600834*arcsec_nV_to_ray**2]
    TTM1 = [TTM1_ray1, TTM1_ray2, TTM1_ray3, TTM1_ray4]
    TTM2 = [TTM2_ray1, TTM2_ray2, TTM2_ray3, TTM2_ray4]
    return TTM1, TTM2

print(generate_TTM_nV(20)[0])
print(generate_TTM_nV(20)[1])