import time
import json
from os import makedirs
from pprint import pprint
from shutil import copy as cp

import numpy as np
from numpy import (cross, dot, pi as π, matrix, tan)
from numpy.linalg import norm as L2norm
from scipy.misc import imshow, imsave

EPSILON = 1e-6

def vector(v):
    return np.float64(v)

def normalize(v):
    return vector(v) / L2norm(v)

def data2rad(s):
    # data['F'] := <number> <deg | rad>
    t = s.split(' ')
    return float(t[0]) if t[1] == 'rad' else (float(t[0]) * π / 180)

def rotateMatFromNegZ(vec):
    # http://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    v1 = vector([0, 0, -1])
    v2 = normalize(vec)
    v = cross(v1, v2)
    s = L2norm(v)
    c = dot(v1, v2)

    # too little rotation or too near 180 deg rotation
    if s ** 2 < EPSILON or abs(1 - c) < EPSILON:
        sign = np.sign(c) or 1 # 1 for 0
        return matrix(sign * np.eye(3))

    vx = matrix([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + vx + (vx ** 2) / (1 + c)
    return R

def isRaySphereIntersected(ray, sphere):
    # https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
    c, r = vector(sphere['o']), vector(sphere['r'])
    o, l = ray['o'], normalize(ray['d'])
    oc = o - c
    _1 = dot(l, oc) ** 2
    _2 = L2norm(oc) ** 2
    D = _1 - _2 + r ** 2
    if D < 0:
        return False
    dsmall = -(dot(l, oc)) - D ** 0.5
    return True if dsmall >= 0 else False

def isRayTriangleIntersected(ray, triangle):
    # https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
    v1, v2, v3 = vector(triangle)
    O, D = ray['o'], ray['d']
    e1 = v2 - v1
    e2 = v3 - v1
    P = cross(D, e2)
    det = dot(e1, P)
    if det > -EPSILON and det < EPSILON:
        return False
    inv_det = 1 / det
    T = O - v1
    u = dot(T, P) * inv_det
    if u < 0 or u > 1:
        return False
    Q = cross(T, e1)
    v = dot(D, Q) * inv_det
    if v < 0 or u + v > 1:
        return False
    t = dot(e2, Q) * inv_det
    return t > EPSILON

def isIntersected(ray, objectList):
    spheres = objectList['S']
    for sphere in spheres:
        if isRaySphereIntersected(ray, sphere):
            return True

    triangles = objectList['T']
    for triangle in triangles:
        if isRayTriangleIntersected(ray, triangle):
            return True

    return False

def main():
    # read and parse data
    with open('input.json') as inputFile:
        data = json.load(inputFile)
    pprint(data)

    eyePos = vector(data['E'])
    viewDir = vector(data['V']) # OpenGL coordinate system
    resolution = data['R']
    fov = data2rad(data['F']) # radian
    spheres = data['S']
    triangles = data['T']

    # pre-processing
    rh, rw = resolution['h'], resolution['w']
    depth = (1 / tan(fov / 2)) * np.hypot(rw, rh) / 2 # depth.png
    hrange = -(np.arange(rh) - rh // 2) # from + h/2 to -h/2
    wrange = np.arange(rw) - rw // 2
    canvas = np.zeros((rh, rw))
    objectList = {'S': spheres, 'T': triangles}

    # All vectors will rotate from (.0, .0, -1.0) to viewDir
    Rmatrix = rotateMatFromNegZ(viewDir)

    hr0, wr0 = hrange[0], wrange[0]
    for h in hrange:
        for w in wrange:
            (hi, wi) = (hr0 - h, w - wr0)
            vec = normalize([[w, h, -depth]])
            ray = {'o': vector(eyePos), 'd': (Rmatrix * vec.T).A1}
            canvas[hi][wi] = 1 if isIntersected(ray, objectList) else 0

    # output results
    timeInt = int(time.time())
    pathname = 'results/{}/'.format(timeInt)
    makedirs(pathname)
    imsave(pathname + 'result.png', canvas)
    cp('input.json', pathname)
    imshow(canvas)

if __name__ == '__main__':
    main()
