import json
from pprint import pprint

import numpy as np
from scipy.misc import imshow

def normalize(v):
    return np.array(v) / np.linalg.norm(v)

def data2rad(s):
    """
    data['F'] := <number> <deg|rad>
    """
    t = s.split(' ')
    return float(t[0]) if t[1] == 'rad' else (float(t[0]) * np.pi / 180)

def rotateMatFromNegZ(vec):
    # http://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    v1 = np.array([0.0, 0.0, -1.0])
    v2 = normalize(vec)
    v = np.cross(v1, v2)
    s = np.linalg.norm(v)
    c = np.dot(v1, v2)
    vx = np.matrix([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    R = np.eye(3) + vx + vx * vx * (1 - c) / (s * s)
    return R

def isRaySphereIntersected(ray, sphere):
    # https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
    c, r = sphere['o'], sphere['r']
    o, l = ray['o'], normalize(ray['d'])
    oc = np.subtract(o, c)
    _1 = np.dot(l, oc) ** 2
    _2 = np.linalg.norm(oc) ** 2
    D = _1 - _2 + r ** 2
    if D < 0:
        return False
    dsmall = -(np.dot(l, oc)) - D ** 0.5
    return True if dsmall >= 0 else False

def isIntersected(ray, objectList):
    spheres = objectList['S']
    for sphere in spheres:
        if isRaySphereIntersected(ray, sphere):
            return True

    triangles = objectList['T']
    for triangle in triangles:
        pass

    return False



def main():
    # read and parse data
    with open('input.json') as inputFile:
        data = json.load(inputFile)
    pprint(data)

    eyePos = np.array(data['E'])
    viewDir = np.array(data['V']) # OpenGL coordinate system
    resolution = data['R']
    fov = data2rad(data['F']) # radian
    spheres = data['S']
    triangles = data['T']

    # pre-processing
    rh, rw = resolution['h'], resolution['w']
    depth = (1 / np.tan(fov / 2)) * np.hypot(rw, rh) / 2 # depth.png
    hrange = -(np.arange(rh) - rh//2) # from + h/2 to -h/2
    wrange = np.arange(rw) - rw//2
    canvas = np.zeros((rh, rw))
    objectList = {'S': spheres, 'T': triangles}

    # All vectors will rotate from (.0, .0, -1.0) to viewDir
    Rmatrix = rotateMatFromNegZ(viewDir)

    hr0, wr0 = hrange[0], wrange[0]
    for h in hrange:
        for w in wrange:
            (hi, wi) = (hr0 - h, w - wr0)
            vec = normalize(np.matrix([w, h, -depth]))
            ray = {'o': eyePos, 'd': (Rmatrix * vec.T).T}
            canvas[hi][wi] = 1 if isIntersected(ray, objectList) else 0

    imshow(canvas)

if __name__ == '__main__':
    main()
