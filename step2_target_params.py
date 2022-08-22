import numpy as np

def target_params_step2():
    dot10 = np.cos(np.deg2rad(5/3600))
    ray1_10arcsec = np.array([-np.sqrt((1-dot10**2)/2), dot10, -np.sqrt((1-dot10**2)/2)])
    ray1_10arcsec = ray1_10arcsec/np.linalg.norm(ray1_10arcsec)

    ray2_10arcsec = np.array([np.sqrt((1-dot10**2)/2), dot10, -np.sqrt((1-dot10**2)/2)])
    ray2_10arcsec = ray2_10arcsec/np.linalg.norm(ray2_10arcsec)

    ray3_10arcsec = np.array([-np.sqrt((1-dot10**2)/2), dot10, np.sqrt((1-dot10**2)/2)])
    ray3_10arcsec = ray3_10arcsec/np.linalg.norm(ray3_10arcsec)

    ray4_10arcsec = np.array([np.sqrt((1-dot10**2)/2), dot10, np.sqrt((1-dot10**2)/2)])
    ray4_10arcsec = ray4_10arcsec/np.linalg.norm(ray4_10arcsec)

    target_10arcsec = [ray1_10arcsec, ray2_10arcsec, ray3_10arcsec, ray4_10arcsec]

    dot20 = np.cos(np.deg2rad(10/3600))
    ray1_20arcsec = np.array([-np.sqrt((1-dot20**2)/2), dot20, -np.sqrt((1-dot20**2)/2)])
    ray1_20arcsec = ray1_20arcsec/np.linalg.norm(ray1_20arcsec)

    ray2_20arcsec = np.array([np.sqrt((1-dot20**2)/2), dot20, -np.sqrt((1-dot20**2)/2)])
    ray2_20arcsec = ray2_20arcsec/np.linalg.norm(ray2_20arcsec)

    ray3_20arcsec = np.array([-np.sqrt((1-dot20**2)/2), dot20, np.sqrt((1-dot20**2)/2)])
    ray3_20arcsec = ray3_20arcsec/np.linalg.norm(ray3_20arcsec)

    ray4_20arcsec = np.array([np.sqrt((1-dot20**2)/2), dot20, np.sqrt((1-dot20**2)/2)])
    ray4_20arcsec = ray4_20arcsec/np.linalg.norm(ray4_20arcsec)

    target_20arcsec = [ray1_20arcsec, ray2_20arcsec, ray3_20arcsec, ray4_20arcsec]

    dot30 = np.cos(np.deg2rad(15/3600))
    ray1_30arcsec = np.array([-np.sqrt((1-dot30**2)/2), dot30, -np.sqrt((1-dot30**2)/2)])
    ray1_30arcsec = ray1_30arcsec/np.linalg.norm(ray1_30arcsec)

    ray2_30arcsec = np.array([np.sqrt((1-dot30**2)/2), dot30, -np.sqrt((1-dot30**2)/2)])
    ray2_30arcsec = ray2_30arcsec/np.linalg.norm(ray2_30arcsec)

    ray3_30arcsec = np.array([-np.sqrt((1-dot30**2)/2), dot30, np.sqrt((1-dot30**2)/2)])
    ray3_30arcsec = ray3_30arcsec/np.linalg.norm(ray3_30arcsec)

    ray4_30arcsec = np.array([np.sqrt((1-dot30**2)/2), dot30, np.sqrt((1-dot30**2)/2)])
    ray4_30arcsec = ray4_30arcsec/np.linalg.norm(ray4_30arcsec)

    target_30arcsec = [ray1_30arcsec, ray2_30arcsec, ray3_30arcsec, ray4_30arcsec]

    dot40 = np.cos(np.deg2rad(20/3600))
    ray1_40arcsec = np.array([-np.sqrt((1-dot40**2)/2), dot40, -np.sqrt((1-dot40**2)/2)])
    ray1_40arcsec = ray1_40arcsec/np.linalg.norm(ray1_40arcsec)

    ray2_40arcsec = np.array([np.sqrt((1-dot40**2)/2), dot40, -np.sqrt((1-dot40**2)/2)])
    ray2_40arcsec = ray2_40arcsec/np.linalg.norm(ray2_40arcsec)

    ray3_40arcsec = np.array([-np.sqrt((1-dot40**2)/2), dot40, np.sqrt((1-dot40**2)/2)])
    ray3_40arcsec = ray3_40arcsec/np.linalg.norm(ray3_40arcsec)

    ray4_40arcsec = np.array([np.sqrt((1-dot40**2)/2), dot40, np.sqrt((1-dot40**2)/2)])
    ray4_40arcsec = ray4_40arcsec/np.linalg.norm(ray4_40arcsec)

    target_40arcsec = [ray1_40arcsec, ray2_40arcsec, ray3_40arcsec, ray4_40arcsec]

    return target_10arcsec, target_20arcsec, target_30arcsec, target_40arcsec