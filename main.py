import numpy as np

def raySphereIntersection(ray, sphere):
    oc = np.subtract(ray['o'], sphere['c'])
    _1 = np.power(np.dot(ray['d'], oc), 2)
    _2 = np.power(np.linalg.norm(oc), 2)
    d = _1 - _2 + np.power(sphere['r'], 2)
    return d >= 0

def main():
    eyePos = [0.0, 0.0, 0.0]
    viewDir = [0.0, 0.0, -1.0] # OpenGL coordinate system
    resolution = {'w':800, 'h':600}
    fov = 100.0 * np.pi / 180.0 # rad
    sphere = {'c':[0.0, 1.0, 10.0], 'r': 2.0} # x, y, z, r

    depth = (1 / np.tan(fov / 2)) * np.hypot(resolution['w'], resolution['h'])/2

    ww = (resolution['w'] // 2) * 2
    hh = (resolution['h'] // 2) * 2
    canvas = np.zeros((hh, ww))

    hi = 0
    for h in np.arange(hh//2, -hh//2, -1):
        wi = 0
        for w in np.arange(-ww//2, ww//2, 1):
            vec = [w, h, depth]
            uvec = vec / np.linalg.norm(vec) #normalize
            ray = {'o': eyePos, 'd': uvec}
            canvas[hi][wi] = 1 if raySphereIntersection(ray, sphere) else 0
            wi = wi + 1
        hi = hi + 1

    from scipy import misc
    misc.imshow(canvas)

if __name__ == '__main__':
    main()
