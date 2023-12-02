def degrade(reference, rotation, translation, scale, drop, duplications, noise):
    from numpy import delete
    from coherent_point_drift.geometry import rotationMatrix, rigidXform
    from itertools import chain, repeat

    points = delete(reference, drop, axis=0)
    rotation_matrix = rotationMatrix(*rotation)
    indices = chain.from_iterable(repeat(i, n) for i, n in enumerate(duplications))
    return rigidXform(points, rotation_matrix, translation, scale)[list(indices)] + noise

def generateDegradation(args, seed):
    from numpy.random import RandomState
    from numpy.linalg import norm

    rs = RandomState(seed)

    if args.D == 2:
        rotation = (rs.uniform(*args.rotate),)
    if args.D == 3:
        angle = rs.uniform(*args.rotate)
        axis = rs.uniform(size=3)
        axis = axis/norm(axis)
        rotation = angle, axis
    translation = rs.uniform(*args.translate, size=args.D)
    scale = rs.uniform(*args.scale)
    if args.drop[0] == args.drop[1]:
        ndrops = args.drop[0]
    else:
        ndrops = rs.randint(*sorted(args.drop))
    drops = rs.choice(range(args.N), size=ndrops, replace=False)
    duplications = rs.choice(range(args.duplicate[0], args.duplicate[1] + 1), size=args.N - ndrops)
    noise = rs.uniform(*args.noise) * rs.randn(sum(duplications), args.D)

    return rotation, translation, scale, drops, duplications, noise

def loadAll(f):
    from pickle import load
    while True:
        try:
            yield load(f)
        except EOFError:
            break
