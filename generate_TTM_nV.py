import numpy as np

def generate_TTM_nV_ans(arcsec_nV_to_ray):
    #arcsec_nV_to_ray = arcsec_ray_to_ray/2
    TTM1_ray1 = [- 0.690534 - 0.00432161*arcsec_nV_to_ray + 0.000355885*arcsec_nV_to_ray**2 - 0.00000884993*arcsec_nV_to_ray**3,
                - 0.718011 + 0.00353096*arcsec_nV_to_ray - 0.000290763*arcsec_nV_to_ray**2 + 0.00000723021*arcsec_nV_to_ray**3,
                - 0.0946425 + 0.00877285*arcsec_nV_to_ray - 0.000722853*arcsec_nV_to_ray**2 + 0.0000179788*arcsec_nV_to_ray**3]
    
    TTM1_ray2 = [- 0.72066 + 0.00280463*arcsec_nV_to_ray - 0.000227365*arcsec_nV_to_ray**2 + 0.00000557226*arcsec_nV_to_ray**3,
                - 0.688922 - 0.00334809*arcsec_nV_to_ray + 0.000271419*arcsec_nV_to_ray**2 - 0.00000665199*arcsec_nV_to_ray**3,
                - 0.0819326 + 0.00579628*arcsec_nV_to_ray - 0.000469891*arcsec_nV_to_ray**2 + 0.000011515*arcsec_nV_to_ray**3]
    
    TTM1_ray3 = [- 0.695435 - 0.00288071*arcsec_nV_to_ray + 0.00023838*arcsec_nV_to_ray**2 - 0.00000595411*arcsec_nV_to_ray**3,
                - 0.714299 + 0.00236119*arcsec_nV_to_ray - 0.000195381*arcsec_nV_to_ray**2 + 0.00000487996*arcsec_nV_to_ray**3,
                0.0821655 - 0.00596745*arcsec_nV_to_ray + 0.000494054*arcsec_nV_to_ray**2 - 0.0000123422*arcsec_nV_to_ray**3]

    TTM1_ray4 =[- 0.716958 + 0.00215076*arcsec_nV_to_ray - 0.000174059*arcsec_nV_to_ray**2 + 0.00000426248*arcsec_nV_to_ray**3,
                - 0.693289 - 0.00254368*arcsec_nV_to_ray + 0.000205856*arcsec_nV_to_ray**2 - 0.00000504114*arcsec_nV_to_ray**3,
                0.0755864 - 0.00437387*arcsec_nV_to_ray + 0.000353986*arcsec_nV_to_ray**2 - 0.00000866806*arcsec_nV_to_ray**3]
    
    TTM2_ray1 = [0.688695 - 0.00440239*arcsec_nV_to_ray + 0.000358081*arcsec_nV_to_ray**2 - 0.00000890604*arcsec_nV_to_ray**3,
                - 0.72552 - 0.00320898*arcsec_nV_to_ray + 0.000260844*arcsec_nV_to_ray**2 - 0.00000648477*arcsec_nV_to_ray**3,
                - 0.0291696 - 0.00942342*arcsec_nV_to_ray + 0.00076654*arcsec_nV_to_ray**2 - 0.0000190691*arcsec_nV_to_ray**3]

    TTM2_ray2 = [0.74959 + 0.00269531*arcsec_nV_to_ray - 0.000214161*arcsec_nV_to_ray**2 + 0.00000524839*arcsec_nV_to_ray**3,
                - 0.661167 + 0.00367408*arcsec_nV_to_ray - 0.000292109*arcsec_nV_to_ray**2 + 0.00000716035*arcsec_nV_to_ray**3,
                - 0.0407833 - 0.00542543*arcsec_nV_to_ray + 0.000431989*arcsec_nV_to_ray**2 - 0.0000105864*arcsec_nV_to_ray**3]
    
    TTM2_ray3 = [0.683966 - 0.00299394*arcsec_nV_to_ray + 0.000243223*arcsec_nV_to_ray**2 - 0.00000607632*arcsec_nV_to_ray**3,
                - 0.728768 - 0.00216006*arcsec_nV_to_ray + 0.000175363*arcsec_nV_to_ray**2 - 0.00000437895*arcsec_nV_to_ray**3,
                0.0424726 + 0.00643281*arcsec_nV_to_ray - 0.000522732*arcsec_nV_to_ray**2 + 0.0000130594*arcsec_nV_to_ray**3]
    
    TTM2_ray4 = [0.753152 + 0.00205436*arcsec_nV_to_ray - 0.000161951*arcsec_nV_to_ray**2 + 0.00000396573*arcsec_nV_to_ray**3,
                - 0.656515 + 0.00281947*arcsec_nV_to_ray - 0.000222423*arcsec_nV_to_ray**2 + 0.00000544803*arcsec_nV_to_ray**3,
                0.0465262 + 0.00414091*arcsec_nV_to_ray - 0.0003273*arcsec_nV_to_ray**2 + 0.00000801486*arcsec_nV_to_ray**3]
    
    TTM1 = [TTM1_ray1, TTM1_ray2, TTM1_ray3, TTM1_ray4]
    TTM2 = [TTM2_ray1, TTM2_ray2, TTM2_ray3, TTM2_ray4]
    return TTM1, TTM2

print(generate_TTM_nV_ans(20)[0])
print(generate_TTM_nV_ans(20)[1])