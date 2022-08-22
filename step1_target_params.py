import numpy as np

def target_params():
    ray1_10arcsec = [
        np.array([1955.500, 88.701, 1.899]),
        np.array([0.9999995711, 0.0006549418, -0.0006549418])]

    ray2_10arcsec = [
        np.array([1955.500, 92.499, 1.899]),
        np.array([0.9999995711, -0.0006549418, -0.0006549418])]

    ray3_10arcsec = [
        np.array([1955.500, 88.701, -1.899]),
        np.array([0.9999995711, 0.0006549418, 0.0006549418])]

    ray4_10arcsec = [
        np.array([1955.500, 92.499, -1.899]),
        np.array([0.9999995711, -0.0006549418, 0.0006549418])]

    target_10arcsec = [ray1_10arcsec, ray2_10arcsec, ray3_10arcsec, ray4_10arcsec]

    ray1_20arcsec = [
        np.array([1955.500, 88.662, 1.938]),
        np.array([0.9999980834, 0.0013844045, -0.0013844045])]

    ray2_20arcsec = [
        np.array([1955.500, 92.534, 1.934]),
        np.array([0.9999980834, -0.0013844045, -0.0013844045])]

    ray3_20arcsec = [
        np.array([1955.500, 88.698, -1.903]),
        np.array([0.9999980834, 0.0013844045, 0.0013844045])]

    ray4_20arcsec = [
        np.array([1955.500, 92.482, -1.881]),
        np.array([0.9999980834, -0.0013844045, 0.0013844045])]

    target_20arcsec = [ray1_20arcsec, ray2_20arcsec, ray3_20arcsec, ray4_20arcsec]

    ray1_30arcsec = [
        np.array([1955.500, 92.179, -1.579]),
        np.array([0.9999949147, 0.0022550561, -0.0022550561])]

    ray2_30arcsec = [
        np.array([1955.500, 89.021, -1.579]),
        np.array([0.9999949147, -0.0022550561, -0.0022550561])]

    ray3_30arcsec = [
        np.array([1955.500, 92.183, 1.583]),
        np.array([0.9999949147, 0.0022550561, 0.0022550561])]

    ray4_30arcsec = [
        np.array([1955.500, 89.016, 1.584]),
        np.array([0.9999949147, -0.0022550561, 0.0022550561])]

    target_30arcsec = [ray1_30arcsec, ray2_30arcsec, ray3_30arcsec, ray4_30arcsec]

    ray1_40arcsec = [
        np.array([1955.500, 92.707, -2.107]),
        np.array([0.9999909374, 0.0030104115, -0.0030104115])]

    ray2_40arcsec = [
        np.array([1955.500, 88.493, -2.107]),
        np.array([0.9999909374, -0.0030104115, -0.0030104115])]

    ray3_40arcsec = [
        np.array([1955.500, 92.707, 2.107]),
        np.array([0.9999909374, 0.0030104115, 0.0030104115])]

    ray4_40arcsec = [
        np.array([1955.500, 88.493, 2.107]),
        np.array([0.9999909374, -0.0030104115, 0.0030104115])]

    target_40arcsec = [ray1_40arcsec, ray2_40arcsec, ray3_40arcsec, ray4_40arcsec]

    return target_10arcsec, target_20arcsec, target_30arcsec, target_40arcsec