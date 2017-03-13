import numpy as np

def normalize(v):
    return np.array(v) / np.linalg.norm(v)

def rotateMatFrom2Vecs(v1, v2):
    # http://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    v1 = normalize(v1)
    v2 = normalize(v2)
    v = np.cross(v1, v2)
    s = np.linalg.norm(v)
    c = np.dot(v1, v2)
    vx = np.matrix([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    R = np.eye(3) + vx + vx * vx * (1 - c) / (s * s)
    return R

def raySphereIntersection(ray, sphere):
    # https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
    oc = np.subtract(ray['o'], sphere['c'])
    _1 = np.power(np.dot(normalize(ray['d']), oc), 2)
    _2 = np.power(np.linalg.norm(oc), 2)
    d = _1 - _2 + np.power(sphere['r'], 2)
    return d >= 0

def main():
    eyePos = [0.0, 0.0, 0.0]
    viewDir = [0.1, 0.1, -1.0] # OpenGL coordinate system
    resolution = {'w':800, 'h':600}
    fov = 100.0 * np.pi / 180.0 # rad
    sphere = {'c':[0.0, 0.0, -10.0], 'r': 2.0} # x, y, z, r

    depth = (1 / np.tan(fov / 2)) * np.hypot(resolution['w'], resolution['h'])/2

    ww = (resolution['w'] // 2) * 2
    hh = (resolution['h'] // 2) * 2
    canvas = np.zeros((hh, ww))

    Rmatrix = rotateMatFrom2Vecs([0.0, 0.0, -1.0], viewDir)

    hi = 0
    for h in np.arange(hh//2, -hh//2, -1):
        wi = 0
        for w in np.arange(-ww//2, ww//2, 1):
            vec = normalize(np.matrix([w, h, -depth]))
            ray = {'o': eyePos, 'd': (Rmatrix * vec.T).T}
            canvas[hi][wi] = 1 if raySphereIntersection(ray, sphere) else 0
            wi = wi + 1
        hi = hi + 1

    from scipy import misc
    misc.imshow(canvas)

if __name__ == '__main__':
    main()
