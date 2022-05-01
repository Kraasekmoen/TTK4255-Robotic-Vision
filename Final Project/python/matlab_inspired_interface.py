import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def show_matched_features(I1, I2, points1, points2, method='falsecolor'):
    """
    Display corresponding feature points. Consistent with Matlab Computer
    Vision Toolbox's showMatchedFeatures, except that method must be either
    'falsecolor' or 'montage', and the images must be of the same resolution
    and grayscale.

    points1 and points2 are Nx2 arrays of corresponding coordinates in I1 and I2.
    See the example usage in example_match_features.py for how to convert a list
    of SIFT KeyPoints to an Nx2 array of coordinates.
    """
    assert I1.shape == I2.shape
    assert I1.ndim == 2, 'I1 and I2 must be grayscale images'
    assert points1.shape == points2.shape
    assert method in ['montage', 'falsecolor'], 'method must be one of "falsecolor" or "montage" but was "%s"' % method
    if method == 'montage':
        plt.imshow(np.concatenate((I1,I2), axis=-1), cmap='gray')
        points2 = points2.copy()
        points2[:,0] += I1.shape[1]
    else:
        plt.imshow(np.stack((I1, I2, I2), axis=-1))
    plt.scatter(points1[:,0], points1[:,1], s=10, linewidths=0.5, facecolors='none', edgecolors='#ff0000', label='Matched points 1')
    plt.scatter(points2[:,0], points2[:,1], s=20, marker='+', linewidths=0.5, color='#00ff00', label='Matched points 2')
    plt.legend()
    plt.plot(
        np.stack((points1[:,0], points2[:,0])),
        np.stack((points1[:,1], points2[:,1])),
        color='#ffff00', linewidth=0.5)
    plt.xticks([])
    plt.yticks([])

def match_features(features1, features2, max_ratio=1.0, unique=False):
    """
    Consistent with Matlab Computer Vision Toolbox's matchFeatures, except
    that it only implements exhaustive ("brute-force") matching, and only
    uses the default distance metric, and does not implement the matching
    threshold check (as an alternative you may use the returned match_metric
    array to implement such thresholds yourself). The text below is copied
    from the Matlab documentation with small modifications.

    ARGUMENTS
    max_ratio:
    Ratio threshold, specified as a scalar ratio value in the range (0,1]. Use
    the max ratio for rejecting ambiguous matches. Increase this value to return
    more matches.

    unique:
    Set this value to true to return only unique matches between features1 and
    features2. When you set unique to false, the function returns all matches
    between features1 and features2. Multiple features in features1 can match
    to one feature in features2. When you set unique to true, the function
    performs a forward-backward match to select a unique match. After matching
    features1 to features2, it matches features2 to features1 and keeps the
    best match. [Note: This is equivalent to the crossCheck functionality in
    the BFMatcher class, but is implemented manually here because OpenCV does
    not let you combine the ratio test together with cross-checking, whereas
    Matlab does allow it.]

    RETURN VALUES
    index_pairs: Indices of corresponding features between the two input
    feature sets, returned as a P-by-2 matrix of P number of indices. Each
    index pair corresponds to a matched feature between the features1 and
    features2 inputs. The first element indexes the feature in features1. The
    second element indexes the matching feature in features2.

    match_metric: Distance between matching features, returned as a P-by-1
    vector. Each ith element in match_metric corresponds to the ith row in the
    index_pairs output matrix.
    """
    index_pairs = []
    match_metric = []

    matcher = cv.BFMatcher()
    matches1to2 = matcher.knnMatch(features1, features2, k=2)
    indices1to2 = []
    for m,n in matches1to2:
        if m.distance <= max_ratio*n.distance:
            indices1to2.append(m.trainIdx)
        else:
            indices1to2.append(None)

    if unique:
        matches2to1 = matcher.match(features2, features1)
        indices2to1 = [m.trainIdx for m in matches2to1]
        for i1,i2 in enumerate(indices1to2):
            if i2 != None and indices2to1[i2] == i1:
                index_pairs.append((i1,i2))
                match_metric.append(matches1to2[i1][0].distance)

    else:
        for i1,i2 in enumerate(indices1to2):
            if i2 != None:
                index_pairs.append((i1,i2))
                match_metric.append(matches1to2[i1][0].distance)

    return np.array(index_pairs), np.array(match_metric)
