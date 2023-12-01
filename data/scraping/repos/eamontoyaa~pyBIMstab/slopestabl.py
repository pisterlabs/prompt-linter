# -*- coding: utf-8 -*-
"""
Module for evaluating the factor of safety against sliding by using the limit
equilibrium method through the General Limit Equilibrium (GLE) method presented
by `Fredlund & Krahn (1977) <https://doi.org/10.1139/t77-045>`_.
"""


# %%
class SlopeStabl:
    """Creates an instance of an object that allows to evaluate the factor of
    safety against sliding of a slope. ::

        SlopeStabl(slices, seedFS=1, Kh=0, maxIter=50, tol=1e-3,
                   interSlcFunc='halfsine', minLambda=-0.6, maxLambda=0.6,
                   nLambda=10)

    Attributes:
        slices (`Slices` object): object that contains the data structure of
            the slices in which the sliding mass has been divided.
        seedFS (`float` or `int`): Initial value of factor of safety for
            starting the iterarive algorithm. ``1`` is the default value.
        lambda_ (`float` or `int`): Factor that multiplies the interlice
            function to determine the interslices horizontal forces. ``0`` is
            the default value.
        Kh (`float`): horizontal seismic coefficient for the pseudostatic
            analysis. Its positive value represents the force is directed out
            of the slope (i.e. in the direction of failure). ``0`` is the
            default value.
        maxIter (`int`): Maximum number of iterations for stoping the
            algorithm in case the tolerance is not reached. ``50`` is the
            default value.
        tol (`float`): Required tolerace to stop the iterations. Is the
            diference between the 2 last values gotten of factor of safety and
            lambda, it means, two tolerances have to be reached. ``1e-3`` is
            the default value.
        interSlcFunc (`str` or 'float'): Interslice function that relates the
            normal interslice forces and the parameter lambda to obtain the
            shear interslice forces. ``halfsine`` is the default value and
            corresponds to Morgenstern and Price method, but a
            constant number may be input, for example ``interSlcFunc=1``,
            corresponds to Spencer method.
        maxLambda (`float`): Maximum value the lambda parameter can get.
            ``0.6`` is the default value.
        nLambda (`float`): Number of value the lambda parameter can get from
            zero to ``maxLambda``. ``6`` is the default value.

slices, seedFS=1, Kh=0, maxIter=50, tol=1e-3,
                 interSlcFunc='halfsine', maxLambda=0.6, nLambda=6
    Note:
        The class ``Slices`` requires
        `numpy <http://www.numpy.org/>`_, `scipy <https://www.scipy.org/>`_,
        `matplotlib <https://matplotlib.org/>`_ and
        `shapely <https://pypi.python.org/pypi/Shapely>`_.

    Examples:
        >>> from numpy import array
        >>> from pybimstab.slope import AnthropicSlope
        >>> from pybimstab.slipsurface import CircularSurface
        >>> from pybimstab.watertable import WaterTable
        >>> from pybimstab.slices import MaterialParameters, Slices
        >>> from pybimstab.slopestabl import SlopeStabl
        >>> slope = AnthropicSlope(slopeHeight=40, slopeDip=[2, 1],
        >>>                        crownDist=60, toeDist=30, depth=20)
        >>> surface = CircularSurface(slopeCoords=slope.coords,
        >>>                           dist1=45.838, dist2=158.726, radius=80)
        >>> material = MaterialParameters(
        >>>     cohesion=600, frictAngle=20, unitWeight=120, wtUnitWeight=62.4)
        >>> watertable = WaterTable(slopeCoords=slope.coords,
                                    watertabDepths=array([[0, 140], [20, 0]]))
        >>> slices = Slices(
        >>>     material=material, slipSurfCoords=surface.coords,
        >>>     slopeCoords=slope.coords, numSlices=50,
        >>>     watertabCoords=watertable.coords, bim=None)
        >>> stabAnalysis = SlopeStabl(slices, seedFS=1, Kh=0)
        >>> stabAnalysis.__dict__.keys()
        dict_keys(['slices', 'Kh', 'seedFS', 'maxIter', 'tol', 'interSlcFunc',
                   'minLambda', 'maxLambda', 'nLambda', 'fsBishop',
                   'fsJanbu', 'fsMoment', 'fsForces', 'lambda_',
                   'adjustment', 'FS'])
        """

    def __init__(self, slices, seedFS=1, Kh=0, maxIter=50, tol=1e-3,
                 interSlcFunc='halfsine', minLambda=-0.6, maxLambda=0.6,
                 nLambda=10):
        '''
        SlopeStabl(slices, seedFS=1, Kh=0, maxIter=50, tol=1e-3,
                   interSlcFunc='halfsine', minLambda=-0.6, maxLambda=0.6,
                   nLambda=10)
        '''
        self.slices = slices
        self.Kh = Kh
        self.seedFS = seedFS
        self.maxIter = maxIter
        self.tol = tol
        self.interSlcFunc = interSlcFunc
        self.minLambda = minLambda
        self.maxLambda = maxLambda
        self.nLambda = nLambda
        # Setting the values of the interslice force function
        self.intersliceForceFunct()
        # Calculating the arms for the moments
        self.calculateArms()
        # Forces that do not vary in each iteration
        self.calculateBasicForces()
        # Get factors of safety for several values of lambda_a
        self.iterateGLE()
        return

    def intersliceForceFunct(self, v=1, u=1):
        '''
        Method for calculating the interslice function which is a component of
        the interslice forces; this is done by using the Equation [11] of
        `Zhu et al (2015) <https://doi.org/10.1139/t04-072>`_, with
        v = u = 1 for a simetric and non-narrowed halfsine function.

        When the object is instanced with the clases with a constant interslice
        function, then, all the values are equal to that constant value.

        Args:
            v (`int` or `float`): shape parameter. Controls the symmetry. ``1``
                is the defaut value.
            u (`int` or `float`): shape parameter. Controls the kurtosis. ``1``
                is the defaut value.

        Returns:
            (`list`): Values of the all insterslice force function values.

        Examples:
            >>> import matplotlib.pyplot as plt
            >>> from pybimstab.slope import AnthropicSlope
            >>> from pybimstab.slipsurface import CircularSurface
            >>> from pybimstab.slices import MaterialParameters, Slices
            >>> from pybimstab.slopestabl import SlopeStabl
            >>> slope = AnthropicSlope(slopeHeight=40, slopeDip=[2, 1],
            >>>                        crownDist=60, toeDist=30, depth=20)
            >>> surface = CircularSurface(slopeCoords=slope.coords,
            >>>                           dist1=45.838, dist2=158.726,
            >>>                           radius=80)
            >>> material = MaterialParameters(cohesion=600, frictAngle=20,
            >>>                               unitWeight=120,
            >>>                               wtUnitWeight=62.4)
            >>> slices = Slices(
            >>>     material=material, slipSurfCoords=surface.coords,
            >>>     slopeCoords=slope.coords, numSlices=50)
            >>> stabAnalysis = SlopeStabl(slices, seedFS=1, Kh=0)
            >>> interslcForceFunc = stabAnalysis.intersliceForceFunct(u=1)
            >>> plt.plot(interslcForceFunc, 'k')

            .. figure:: https://rawgit.com/eamontoyaa/pybimstab/master/examples/figures/slopestabl_interSlcFunct_example.svg
                :alt: slopestabl_interSlcFunct_example

            .. only:: html

                :download:`example script<../examples/figuresScripts/slopestabl_interSlcFunct_example.py>`.
        '''
        from numpy import sin, pi

        # Getting the ends of the slip surface
        a = self.slices.slices[0].terrainCoords[0].min()  # leftmost point
        b = self.slices.slices[-1].terrainCoords[0].max()  # rightmost point
        interslcForceFunc = list()
        # Calculate the interslice functions values (f_i = fL, f_{i-1} = fR)
        for slice_ in self.slices.slices:
            lf = slice_.xMin  # leftmost slice point
            rt = slice_.xMax  # rightmost slice point
            # Evaluating the half-sine function on the sides of the slice
            if self.interSlcFunc == 'halfsine':
                halfSineL = sin(pi * ((lf - a) / (b - a)) ** v) ** u
                halfSineR = sin(pi * ((rt - a) / (b - a)) ** v) ** u
            else:  # if not a half-sine, then use the input constant value
                halfSineL = self.interSlcFunc
                halfSineR = self.interSlcFunc
            interslcForceFunc.append(halfSineL)
            setattr(slice_, 'fL', halfSineL)
            setattr(slice_, 'fR', halfSineR)
        interslcForceFunc.append(halfSineR)
        return interslcForceFunc

    def calculateArms(self):
        '''
        Method for calculating the arms required for getting the momments of
        each slice with respect to a rotation point.

        This function does not return any output, just modifies the structure
        of each slice by setting new attributes.
        '''
        import numpy as np
        from numpy import radians as rad
        from shapely.geometry import LineString

        for n, slice_ in enumerate(self.slices.slices):
            setattr(slice_, 'n', n)
            # Vertical linestring that splits the slice through the cetroid
            xMean = (slice_.xMin + slice_.xMax) / 2
            vertLine = LineString([(xMean, -self.slices.rotationPt[1]),
                                   (xMean, self.slices.rotationPt[1])])

            # Arms for external loads
            loadPt1 = np.array(slice_.terrainLS.intersection(vertLine))
            theta = np.arctan((self.slices.rotationPt[1] - loadPt1[1]) /
                              (self.slices.rotationPt[0] - loadPt1[0])) % np.pi
            alpha = abs(rad(slice_.w) - theta)
            loadPt2RotPt = np.linalg.norm(self.slices.rotationPt - loadPt1)
            proy = loadPt2RotPt * abs(np.cos(alpha))
            d = (loadPt2RotPt ** 2 - proy ** 2) ** 0.5
            if rad(slice_.w) < theta:
                d *= -1
            setattr(slice_, 'd', d)

            # Arms for normal loads at the base
            loadPt2 = slice_.slipSurfLS.intersection(vertLine)
            if loadPt2.type is not 'Point':
                loadPt2 = [xMean, loadPt1[1] - slice_.midHeight]
            loadPt2 = np.array(loadPt2)
            theta = np.arctan((self.slices.rotationPt[1] - loadPt2[1]) /
                              (self.slices.rotationPt[0] - loadPt2[0])) % np.pi
            alpha = abs(0.5*np.pi - rad(slice_.alpha) - theta)
            loadPt2RotPt = np.linalg.norm(self.slices.rotationPt - loadPt2)
            proy = loadPt2RotPt * abs(np.cos(alpha))
            f = (loadPt2RotPt ** 2 - proy ** 2) ** 0.5
            if 0.5*np.pi - rad(slice_.alpha) < theta:
                f *= -1
            setattr(slice_, 'f', f)
            setattr(slice_, 'R', proy)  # Arm for the mobilized shear force

            # Arms for horizontal seismic force
            e = self.slices.rotationPt[1] - (loadPt2[1] + 0.5*slice_.midHeight)
            setattr(slice_, 'e', e)

            # Arms for the weight of the slice
            x = self.slices.rotationPt[0] - xMean
            setattr(slice_, 'x', x)
        return

    def calculateBasicForces(self):
        '''
        Method for calculating the forces that do not vary in each iteration or
        lambda value.

        This function does not return any output, just modifies the structure
        of each slice by setting new attributes.
        '''

        for slice_ in self.slices.slices:
            if self.slices.bim is not None:
                # blocks and matrix areas to get slice weight
                blocksArea = slice_.numBlocks * slice_.localBIM.tileSize**2
                mtxArea = slice_.area - blocksArea
                weight = slice_.material.blocksUnitWeight * \
                    blocksArea + mtxArea*slice_.material.unitWeight
            else:
                weight = slice_.area * slice_.material.unitWeight
            setattr(slice_, 'weight', weight)

            # Average water pressure (mu) and the resultant water force (U)
            mu = slice_.material.wtUnitWeight * slice_.midWatTabHeight
            setattr(slice_, 'mu', mu)
            U = mu * slice_.l
            setattr(slice_, 'U', U)

            # Setting interslices forces equal to zero
            setattr(slice_, 'Xl', 0)
            setattr(slice_, 'Xr', 0)
            setattr(slice_, 'El', 0)
            setattr(slice_, 'Er', 0)
        return

    def calculateNormalForce(self, seedFS, fellenius=False):
        '''
        Method for calculating the normal force to the base; this is done by
        using the Equation of section 14.6 of `GeoSlope (2015) <http://downloads.geo-slope.com/geostudioresources/books/8/15/slope%20modeling.pdf>`_

        Since the normal forces are updated with each iteration, is necessary
        to input a factor of safety as a seed.

        Args:
            seedFS (`int` or `float`): Seed factor of safety.

        Returns:
            (`list`): Values of all the normal forces at the slice's bases

        Examples:
            >>> from numpy import array
            >>> import matplotlib.pyplot as plt
            >>> from pybimstab.slope import AnthropicSlope
            >>> from pybimstab.slipsurface import CircularSurface
            >>> from pybimstab.watertable import WaterTable
            >>> from pybimstab.slices import MaterialParameters, Slices
            >>> from pybimstab.slopestabl import SlopeStabl
            >>> slope = AnthropicSlope(slopeHeight=40, slopeDip=[2, 1],
            >>>                        crownDist=60, toeDist=30, depth=20)
            >>> surface = CircularSurface(slopeCoords=slope.coords,
            >>>                           dist1=45.838, dist2=158.726,
            >>>                           radius=80)
            >>> material = MaterialParameters(cohesion=600, frictAngle=20,
            >>>                               unitWeight=120,
            >>>                               wtUnitWeight=62.4)
            >>> slices = Slices(
            >>>     material=material, slipSurfCoords=surface.coords,
            >>>     slopeCoords=slope.coords, numSlices=5)
            >>> stabAnalysis = SlopeStabl(slices, seedFS=1, Kh=0)
            >>> stabAnalysis.calculateNormalForce(stabAnalysis.FS['fs'])
            [45009.409630951726, 68299.77910530512, 70721.13554871723,
             57346.7578530581, 22706.444365285253]
        '''
        from numpy import sin, cos, tan
        from numpy import radians as rad

        listP = list()
        for slice_ in self.slices.slices:
            # Calculating the normal force 'P' at the base of the slice_.
            c = slice_.material.cohesion
            phi = rad(slice_.material.frictAngle)
            if fellenius:
                P = slice_.weight * cos(rad(slice_.alpha)) - \
                    self.Kh * slice_.weight * sin(rad(slice_.alpha))
            else:
                if seedFS == 0:
                    seedFS = 1
                mAlpha = cos(rad(slice_.alpha)) + sin(rad(slice_.alpha)) * \
                    tan(phi) / seedFS
                # Eq. [16] of Fredlund & Kranh (1977) does not work for now
                # Eq. gotten from the section 14.6 of GEO-SLOPE (2015)
                P = (slice_.weight + slice_.Xr - slice_.Xl -
                     (c * slice_.l - slice_.U * tan(phi)) *
                     sin(rad(slice_.alpha)) / seedFS +
                     slice_.extL * sin(rad(slice_.w))) / mAlpha
            setattr(slice_, 'P', P)
            listP.append(P)
        return listP

    def getFm(self, seedFS, lambda_=0, fellenius=False):
        '''
        Method for getting the factor of safety with respect to the moments
        equilimrium; this is done by using the Equation [22] of
        `Fredlund & Krahn (1977) <https://doi.org/10.1139/t77-045>`_.

        Since the factor of safety is updated with each iteration, is necessary
        to input a factor of safety as a seed and the current value of lambda
        to relate the interslice normal force and the interslice force function
        with respect to the interslice shear force (Eq. [16] of
        `Fredlund & Krahn (1977) <https://doi.org/10.1139/t77-045>`_).

        Args:
            seedFS (`int` or `float`): Seed factor of safety.
            lambda_ (`int` or `float`): Seed value of lambda. ``0`` is the
                default value.

        Returns:
            (`dict`): Dictionary with the value of the factor of safety and a\
                tuple with the boolean that indicated if the toleranfe was\
                reached and the number of the iteration.

        Examples:
            >>> # Example Case 1 - Fig. 9 (Fredlund & Krahn, 1977)
            >>> from pybimstab.slope import AnthropicSlope
            >>> from pybimstab.slipsurface import CircularSurface
            >>> from pybimstab.slices import MaterialParameters, Slices
            >>> from pybimstab.slopestabl import SlopeStabl
            >>> slope = AnthropicSlope(slopeHeight=40, slopeDip=[2, 1],
            >>>                        crownDist=60, toeDist=30, depth=20)
            >>> surface = CircularSurface(slopeCoords=slope.coords,
            >>>                           dist1=45.838, dist2=158.726,
            >>>                           radius=80)
            >>> material = MaterialParameters(cohesion=600, frictAngle=20,
            >>>                               unitWeight=120,
            >>>                               wtUnitWeight=62.4)
            >>> slices = Slices(
            >>>     material=material, slipSurfCoords=surface.coords,
            >>>     slopeCoords=slope.coords, numSlices=50,
            >>>     watertabCoords=None, bim=None)
            >>> stabAnalysis = SlopeStabl(slices, seedFS=1, Kh=0, minLambda=0,
            >>>                           interSlcFunc=1, nLambda=10)
            >>> stabAnalysis.getFm(stabAnalysis.FS['fs'],
            >>>                    stabAnalysis.FS['lambda'])
            (2.0750390044795854, True)
        '''
        from numpy import tan
        from numpy import radians as rad

        # Doing the iteration
        toleraceReached = False
        fs = [seedFS]
        for i in range(self.maxIter):
            # Calculating the normal force at each base slice_.
            self.calculateNormalForce(fs[-1], fellenius)
            num = 0
            den1, den2, den3, den4 = 0, 0, 0, 0
            for slice_ in self.slices.slices:
                c = slice_.material.cohesion
                phi = rad(slice_.material.frictAngle)
                num += c * slice_.l * slice_.R + \
                    (slice_.P - slice_.U) * slice_.R * tan(phi)
                den1 += slice_.weight * slice_.x
                den2 += slice_.P * slice_.f
                den3 += slice_.extL * slice_.d
                den4 += self.Kh * slice_.weight * slice_.e
            fs.append(num / (den1 - den2 + den3 + den4))
            if fellenius:
                break

            # Recalculating the interslice forces
            self.intersliceForces(fs[-1], lambda_)
            self.calculateNormalForce(fs[-1])
            self.intersliceForces(fs[-1], lambda_)

            # Verifying if the tolerance is reached
            if i > 5 and all((fs[-1] - fs[-2], fs[-2] - fs[-3])) <= self.tol:
                toleraceReached = True
                break
        return fs[-1], toleraceReached

    def getFf(self, seedFS, lambda_=0):
        '''
        Method for getting the factor of safety with respect to the forces
        equilimrium; this is done by using the Equation [23] of
        `Fredlund & Krahn (1977) <https://doi.org/10.1139/t77-045>`_.

        Since the factor of safety is updated with each iteration, is necessary
        to input a factor of safety as a seed and the current value of lambda
        to relate the interslice normal force and the interslice force function
        with respect to the interslice shear force (Eq. [16] of
        `Fredlund & Krahn (1977) <https://doi.org/10.1139/t77-045>`_).

        Args:
            seedFS (`int` or `float`): Seed factor of safety.
            lambda_ (`int` or `float`): Seed value of lambda. ``0`` is the
                default value.

        Returns:
            (`dict`): Dictionary with the value of the factor of safety and a\
                tuple with the boolean that indicated if the toleranfe was\
                reached and the number of the iteration.

        Examples:
            >>> # Example Case 1 - Fig. 9 (Fredlund & Krahn, 1977)
            >>> from pybimstab.slope import AnthropicSlope
            >>> from pybimstab.slipsurface import CircularSurface
            >>> from pybimstab.slices import MaterialParameters, Slices
            >>> from pybimstab.slopestabl import SlopeStabl
            >>> slope = AnthropicSlope(slopeHeight=40, slopeDip=[2, 1],
            >>>                        crownDist=60, toeDist=30, depth=20)
            >>> surface = CircularSurface(slopeCoords=slope.coords,
            >>>                           dist1=45.838, dist2=158.726,
            >>>                           radius=80)
            >>> material = MaterialParameters(cohesion=600, frictAngle=20,
            >>>                               unitWeight=120,
            >>>                               wtUnitWeight=62.4)
            >>> slices = Slices(
            >>>     material=material, slipSurfCoords=surface.coords,
            >>>     slopeCoords=slope.coords, numSlices=50,
            >>>     watertabCoords=None, bim=None)
            >>> stabAnalysis = SlopeStabl(slices, seedFS=1, Kh=0, minLambda=0,
            >>>                           interSlcFunc=1, nLambda=10)
            >>> stabAnalysis.getFf(stabAnalysis.FS['fs'],
            >>>                    stabAnalysis.FS['lambda'])
            (2.0741545445738296, True)
        '''
        from numpy import tan, cos, sin
        from numpy import radians as rad

        # Doing the iteration
        toleraceReached = False
        fs = [seedFS]
        for i in range(self.maxIter):
            # Calculating the normal force at each base slice_.
            self.calculateNormalForce(fs[-1])
            num = 0
            den1, den2, den3 = 0, 0, 0
            for slice_ in self.slices.slices:
                c = slice_.material.cohesion
                phi = rad(slice_.material.frictAngle)
                num += c * slice_.width + (slice_.P - slice_.U) * tan(phi) * \
                    cos(rad(slice_.alpha))
                den1 += slice_.P * sin(rad(slice_.alpha))
                den2 += slice_.extL * cos(rad(slice_.w))
                den3 += self.Kh * slice_.weight
            fs.append(num / (den1 - den2 + den3))

            # Recalculating the interslice forces
            self.intersliceForces(fs[-1], lambda_)
            self.calculateNormalForce(fs[-1])
            self.intersliceForces(fs[-1], lambda_)

            # Verifying if the tolerance is reached
            if i > 5 and all((fs[-1] - fs[-2], fs[-2] - fs[-3])) <= self.tol:
                toleraceReached = True
                break
        return fs[-1], toleraceReached

    def intersliceForces(self, seedFS, lambda_):
        '''
        Method for getting the shear and normal interslice forces; this is
        done by using the Equation of section 14.8 of
        `GeoSlope (2015) <http://downloads.geo-slope.com/geostudioresources/books/8/15/slope%20modeling.pdf>`_
        for the rigth normal force and the Equation [18] of
        `Fredlund & Krahn (1977) <https://doi.org/10.1139/t77-045>`_ for the
        shear force.

        Since the interslice forces are updated with each iteration, is
        necessary to input a factor of safety as a seed and the current value
        of lambda to relate the interslice normal force and the interslice
        force function with respect to the interslice shear force (Eq. [20] of
        `Fredlund & Krahn (1977) <https://doi.org/10.1139/t77-045>`_).

        Args:
            seedFS (`int` or `float`): Seed factor of safety.
            lambda_ (`int` or `float`): Seed value of lambda. ``0`` is the
                default value.

        Returns:
            (`tuple`): tuple with the interslice forces. the first element\
                contains the normal interslice forces and the second contains\
                the shear interslice forces.

        Examples:
            >>> from numpy import array
            >>> import matplotlib.pyplot as plt
            >>> from pybimstab.slope import AnthropicSlope
            >>> from pybimstab.slipsurface import CircularSurface
            >>> from pybimstab.watertable import WaterTable
            >>> from pybimstab.slices import MaterialParameters, Slices
            >>> from pybimstab.slopestabl import SlopeStabl
            >>> slope = AnthropicSlope(slopeHeight=40, slopeDip=[2, 1],
            >>>                        crownDist=60, toeDist=30, depth=20)
            >>> surface = CircularSurface(slopeCoords=slope.coords,
            >>>                           dist1=45.838, dist2=158.726,
            >>>                           radius=80)
            >>> material = MaterialParameters(cohesion=600, frictAngle=20,
            >>>                               unitWeight=120,
            >>>                               wtUnitWeight=62.4)
            >>> slices = Slices(
            >>>     material=material, slipSurfCoords=surface.coords,
            >>>     slopeCoords=slope.coords, numSlices=5)
            >>> stabAnalysis = SlopeStabl(slices, seedFS=1, Kh=0)
            >>> stabAnalysis.intersliceForces(stabAnalysis.FS['fs'],
            >>>                               stabAnalysis.FS['lambda'])
            ([0, -24561.260979675248, -42085.32887504204, -38993.844201424305,
              -18464.723052348225, -61.4153504520018],
             [0, -5511.202498703704, -15279.673506543182, -14157.266298947989,
              -4143.22489013017, -2.8712090198929304e-15])
        '''
        from numpy import tan, cos, sin
        from numpy import radians as rad

        self.slices.slices[0].El, self.slices.slices[0].Xl = 0, 0
        forcesE = [self.slices.slices[0].El]
        forcesX = [self.slices.slices[0].Xl]
        for i in range(self.slices.numSlices):
            slice_ = self.slices.slices[i]
            c = slice_.material.cohesion
            phi = rad(slice_.material.frictAngle)
            Sm = (c * slice_.l + (slice_.P - slice_.U) * tan(phi))/seedFS
            setattr(slice_, 'Sm', Sm)
            # Eq. [19] of Fredlund & Kranh (1977) does not work for now
#            slice_.Er = slice_.El + (slice_.weight - slice_.Xr + slice_.Xl) *\
#                tan(rad(slice_.alpha)) - Sm / cos(rad(slice_.alpha)) + \
#                self.Kh * slice_.weight
            # Eq. gotten from the section 14.8 of GEO-SLOPE (2015)
            slice_.Er = slice_.El - slice_.P * sin(rad(slice_.alpha)) + \
                Sm * cos(rad(slice_.alpha)) - self.Kh * slice_.weight + \
                slice_.extL * cos(rad(slice_.w))
            slice_.Xr = slice_.Er * lambda_ * slice_.fR
            if i < self.slices.numSlices - 1:
                nextSlice = self.slices.slices[i+1]
                nextSlice.El = -1 * slice_.Er
                nextSlice.Xl = -1 * slice_.Xr
            elif i == self.slices.numSlices-1:
                slice_.Er, slice_.Xr = 0, 0
            forcesE.append(slice_.Er)
            forcesX.append(slice_.Xr)
        return (forcesE, forcesX)

    def iterateGLE(self):
        '''
        Method for getting the factor of safety against sliding through
        the algorithm of the General Limit Equilibrium (GLE) proposed by
        `Fredlund & Krahn (1977) <https://doi.org/10.1139/t77-045>`_).

        Returns:
            (`tuple` or `None`): factor of safety against sliding is the\
                solution exists.

        Examples:
            >>> from numpy import array
            >>> import matplotlib.pyplot as plt
            >>> from pybimstab.slope import AnthropicSlope
            >>> from pybimstab.slipsurface import CircularSurface
            >>> from pybimstab.watertable import WaterTable
            >>> from pybimstab.slices import MaterialParameters, Slices
            >>> from pybimstab.slopestabl import SlopeStabl
            >>> slope = AnthropicSlope(slopeHeight=40, slopeDip=[2, 1],
            >>>                        crownDist=60, toeDist=30, depth=20)
            >>> surface = CircularSurface(slopeCoords=slope.coords,
            >>>                           dist1=45.838, dist2=158.726,
            >>>                           radius=80)
            >>> material = MaterialParameters(cohesion=600, frictAngle=20,
            >>>                               unitWeight=120,
            >>>                               wtUnitWeight=62.4)
            >>> slices = Slices(
            >>>     material=material, slipSurfCoords=surface.coords,
            >>>     slopeCoords=slope.coords, numSlices=5)

            >>> stabAnalysis = SlopeStabl(slices, seedFS=1, Kh=0)
            >>> stabAnalysis.iterateGLE()
            {'fs': 2.0258090954552275, 'lambda': 0.38174822248691215}

            >>> stabAnalysis = SlopeStabl(slices, seedFS=1, Kh=0, nLambda=0)
            >>> stabAnalysis.iterateGLE()
            {'fsBishop': 2.0267026043637175, 'fsJanbu': 1.770864711650081}
        '''
        import numpy as np
        from pybimstab.tools import getIntersect
        from pybimstab.smoothcurve import SmoothCurve

        if self.nLambda > 0:
            # Getting the values of lambda to iterate GLE
            lambdaVal = np.unique(list(np.linspace(
                    self.minLambda, self.maxLambda, self.nLambda)) + [0])

            # Iteration for moments
            fsFellenius, tol = self.getFm(self.seedFS, lambda_=0,
                                          fellenius=True)
            setattr(self, 'fsFellenius', fsFellenius)
            fsBishop, tol = self.getFm(self.seedFS, lambda_=0)  # Moment equil.
            setattr(self, 'fsBishop', fsBishop)
            fsMoment, tolM = [fsBishop], list()
            for lambda_ in lambdaVal:
                fsM, tol = self.getFm(fsMoment[-1], lambda_)
                if max(fsM, fsMoment[-1]) / min(fsM, fsMoment[-1]) > 1.5:
                    tol = False
                fsMoment.append(fsM)
                tolM.append(tol)
                self.intersliceForces(fsMoment[-1], lambda_)
                self.calculateNormalForce(fsMoment[-1])
            fsMoment.pop(0)

            for slice_ in self.slices.slices:
                    slice_.Er, slice_.El, slice_.Xr, slice_.Xl = 0, 0, 0, 0

            # Iteration for forces
            fsJanbu, tol = self.getFf(self.seedFS, lambda_=0)  # Force eq.
            setattr(self, 'fsJanbu', fsJanbu)
            fsForces, tolF = [fsJanbu], list()
            for lambda_ in lambdaVal:
                fsF, tol = self.getFf(fsForces[-1], lambda_)
                if max(fsF, fsForces[-1]) / min(fsF, fsForces[-1]) > 1.5:
                    tol = False
                fsForces.append(fsF)
                tolF.append(tol)
                self.intersliceForces(fsMoment[-1], lambda_)
                self.calculateNormalForce(fsMoment[-1])
            fsForces.pop(0)

            # Creating the attributes
            idx2interp = np.where(tolM and tolF)
            setattr(self, 'fsMoment', list(np.array(fsMoment)[idx2interp]))
            setattr(self, 'fsForces', list(np.array(fsForces)[idx2interp]))
            setattr(self, 'lambda_', list(np.array(lambdaVal)[idx2interp]))

            # Get intersection of factors of safety
            momentLine = SmoothCurve(
                    x=self.lambda_, y=self.fsMoment, k=3, n=100)
            forcesLine = SmoothCurve(
                    x=self.lambda_, y=self.fsForces, k=3, n=100)
            x, momentsY = momentLine.smoothing
            forcesY = forcesLine.smoothing[1]
            setattr(self, 'adjustment', (x, momentsY, forcesY))
            intersect = getIntersect(x=x, y1=momentsY, y2=forcesY)
            if intersect is None:
                root, fs = None, None
                setattr(self, 'FS', {'fs': None, 'lambda': None})
            else:
                root, fs = intersect
                setattr(self, 'FS', {'fs': fs, 'lambda': root})

            # Slices forces when full equilibrium is found (lambda=root)
            if fs is not None:
                self.getFf(fs, root)
            return {'fs': fs, 'lambda': root}
        else:
            fsFellenius, tol = self.getFm(self.seedFS, lambda_=0,
                                          fellenius=True)
            setattr(self, 'fsFellenius', fsFellenius)
            fsBishop, tol = self.getFm(self.seedFS, lambda_=0)
            setattr(self, 'fsBishop', fsBishop)
            fsJanbu, tol = self.getFf(self.seedFS, lambda_=0)
            setattr(self, 'fsJanbu', fsJanbu)
            return {'fsBishop': fsBishop, 'fsJanbu': fsJanbu}

    def plot(self):
        '''Method for generating a graphic of the slope stability analysis,
        including the plot of the convergences

        Returns:
            (`matplotlib.figure.Figure`): object with the matplotlib structure\
                of the plot. You might use it to save the figure for example.

        Examples:
            >>> # Example Case 1 - Fig. 9 (Fredlund & Krahn, 1977)
            >>> from pybimstab.slope import AnthropicSlope
            >>> from pybimstab.slipsurface import CircularSurface
            >>> from pybimstab.slices import MaterialParameters, Slices
            >>> from pybimstab.slopestabl import SlopeStabl
            >>> slope = AnthropicSlope(slopeHeight=40, slopeDip=[2, 1],
            >>>                        crownDist=60, toeDist=30, depth=20)
            >>> surface = CircularSurface(slopeCoords=slope.coords,
            >>>                           dist1=45.838, dist2=158.726,
            >>>                           radius=80)
            >>> material = MaterialParameters(cohesion=600, frictAngle=20,
            >>>                               unitWeight=120,
            >>>                               wtUnitWeight=62.4)
            >>> slices = Slices(
            >>>     material=material, slipSurfCoords=surface.coords,
            >>>     slopeCoords=slope.coords, numSlices=50,
            >>>     watertabCoords=None, bim=None)
            >>> stabAnalysis = SlopeStabl(slices, seedFS=1, Kh=0, minLambda=0,
            >>>                           interSlcFunc=1, nLambda=10)
            >>> fig = stabAnalysis.plot()

            .. figure:: https://rawgit.com/eamontoyaa/pybimstab/master/examples/figures/slopestabl_example1.svg
                :alt: slopestabl_example1

            .. only:: html

                :download:`example script<../examples/figuresScripts/slopestabl_example1.py>`.

            >>> # Example Case 5 (Fredlund & Krahn, 1977)
            >>> from numpy import array
            >>> from pybimstab.slope import AnthropicSlope
            >>> from pybimstab.slipsurface import CircularSurface
            >>> from pybimstab.watertable import WaterTable
            >>> from pybimstab.slices import MaterialParameters, Slices
            >>> from pybimstab.slopestabl import SlopeStabl
            >>> slope = AnthropicSlope(slopeHeight=40, slopeDip=[2, 1],
            >>>                        crownDist=60, toeDist=30, depth=20)
            >>> surface = CircularSurface(slopeCoords=slope.coords,
            >>>                           dist1=45.838, dist2=158.726,
            >>>                           radius=80)
            >>> material = MaterialParameters(cohesion=600, frictAngle=20,
            >>>                               unitWeight=120,
            >>>                               wtUnitWeight=62.4)
            >>> watertable = WaterTable(slopeCoords=slope.coords,
            >>>                         watertabDepths=array([[0, 140],
            >>>                                               [20, 0]]))
            >>> slices = Slices(
            >>>     material=material, slipSurfCoords=surface.coords,
            >>>     slopeCoords=slope.coords, numSlices=50,
            >>>     watertabCoords=watertable.coords, bim=None)
            >>> stabAnalysis = SlopeStabl(slices, seedFS=1, Kh=0, minLambda=0)
            >>> fig = stabAnalysis.plot()

            .. figure:: https://rawgit.com/eamontoyaa/pybimstab/master/examples/figures/slopestabl_example2.svg
                :alt: slopestabl_example2

            .. only:: html

                :download:`example script<../examples/figuresScripts/slopestabl_example2.py>`.

            >>> from numpy import array
            >>> from pybimstab.slope import NaturalSlope
            >>> from pybimstab.watertable import WaterTable
            >>> from pybimstab.bim import BlocksInMatrix
            >>> from pybimstab.slipsurface import CircularSurface
            >>> from pybimstab.slipsurface import TortuousSurface
            >>> from pybimstab.slices import MaterialParameters, Slices
            >>> from pybimstab.slopestabl import SlopeStabl
            >>> terrainCoords = array(
            >>>     [[-2.49, 0.1, 1.7, 3.89, 5.9, 8.12, 9.87, 13.29, 20.29,
            >>>       21.43, 22.28, 23.48, 24.65, 25.17],
            >>>      [18.16, 17.88, 17.28, 15.73, 14.31, 13.58, 13, 3.61, 3.61,
            >>>       3.32, 2.71, 2.23, 1.21, 0.25]])
            >>> slope = NaturalSlope(terrainCoords)
            >>> bim = BlocksInMatrix(slopeCoords=slope.coords, blockProp=0.2,
            >>>                      tileSize=0.35, seed=3210)
            >>> watertabDepths = array([[0, 5, 10, 15],
            >>>                         [8, 7, 3, 0]])
            >>> watertable = WaterTable(slopeCoords=slope.coords,
            >>>                         watertabDepths=watertabDepths,
            >>>                         smoothFactor=3)
            >>> preferredPath = CircularSurface(
            >>>     slopeCoords=slope.coords, dist1=5, dist2=15.78, radius=20)
            >>> surface = TortuousSurface(
            >>>     bim, dist1=4, dist2=15.5, heuristic='euclidean',
            >>>     reverseLeft=False, reverseUp=False, smoothFactor=2,
            >>>     preferredPath=preferredPath.coords, prefPathFact=2)
            >>> material = MaterialParameters(
            >>>     cohesion=15, frictAngle=23, unitWeight=17,
            >>>     blocksUnitWeight=21, wtUnitWeight=9.8)
            >>> slices = Slices(
            >>>     material=material, slipSurfCoords=surface.coords,
            >>>     slopeCoords=slope.coords, numSlices=20,
            >>>     watertabCoords=watertable.coords, bim=bim)
            >>> stabAnalysis = SlopeStabl(slices, seedFS=1, Kh=0, nLambda=13,
            >>>                           minLambda=0)
            >>> fig = stabAnalysis.plot()

            .. figure:: https://rawgit.com/eamontoyaa/pybimstab/master/examples/figures/slopestabl_example3.svg
                :alt: slopestabl_example3

            .. only:: html

                :download:`example script<../examples/figuresScripts/slopestabl_example3.py>`.
        '''
        import numpy as np
        from matplotlib import pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap as newcmap
        from matplotlib import gridspec

        # Variables to control the color map and its legend
        if self.slices.bim is not None:
            if np.any(self.slices.bim.grid == -1):
                cmap = newcmap.from_list('BIMcmap',
                                         ['white', 'lightgray', 'black'], 3)
                ticks = [-1+0.333, 0, 1-0.333]
                ticksLabels = ['None', 'Matrix', 'Blocks']
            else:
                cmap = newcmap.from_list('BIMcmap', ['lightgray', 'black'], 2)
                ticks = [0.25, 0.75]
                ticksLabels = ['Matrix', 'Blocks']
        # Plot body
        if self.nLambda > 0:
            fig = plt.figure(figsize=(9, 4.5))
            gs = gridspec.GridSpec(1, 2, width_ratios=[1, 3])
            ax1 = plt.subplot(gs[1])
            ax2 = plt.subplot(gs[0])
        else:
            fig = plt.figure()
            ax1 = fig.add_subplot(111)

        # # Subfigure 1
        if self.slices.bim is not None:
            bar = ax1.pcolormesh(
                    self.slices.bim.xCells, self.slices.bim.yCells,
                    self.slices.bim.grid, cmap=cmap)
            # Configuring the colorbar
            bar = plt.colorbar(bar, ax=ax1, ticks=ticks, pad=0.01,
                               shrink=0.15, aspect=3)
            bar.ax.set_yticklabels(ticksLabels, fontsize='small')
        for slice_ in self.slices.slices:  # Plotting each slice
            ax1.plot(*slice_.coords, ':r', lw=0.5)
        ax1.plot(*self.slices.slipSurfCoords, '-r')
        ax1.plot(*self.slices.rotationPt, '.r',
                 label='$f_\\mathrm{s\ (Fellenius)} = ' +
                 str(round(self.fsFellenius, 3)) +
                 '$\n$f_\\mathrm{s\ (Bishop\ simp.)} = ' +
                 str(round(self.fsBishop, 3)) +
                 '$\n$f_\\mathrm{s\ (Janbu\ simp.)} = ' +
                 str(round(self.fsJanbu, 3)) + '$')
        ax1.plot(*self.slices.slopeCoords, '-k')
        if self.slices.watertabCoords is not None:
            ax1.plot(*self.slices.watertabCoords, 'deepskyblue', lw=0.9)
        # Plot settings
        ax1.grid(True, ls='--', lw=0.5)
        ax1.axis('equal')
        ax1.tick_params(labelsize='x-small')

        # # Subfigure 2
        if self.nLambda > 0:
            ax2.plot(self.adjustment[0], self.adjustment[1], '-k', lw=0.5)
            ax2.plot(self.lambda_, self.fsMoment, 'vk', ms=3.5)
            ax2.plot(self.adjustment[0], self.adjustment[2], '-k', lw=0.5)
            ax2.plot(self.lambda_, self.fsForces, 'ok', ms=3.5)
            if self.FS['lambda'] is not None:
                ax2.plot(self.FS['lambda'], self.FS['fs'], '*r', ms=7)
            # Plot settings
            ax2.tick_params(labelsize='x-small')
            ax2.grid(True, ls='--', lw=0.5)
            ax2Ticks = np.arange(round(min(self.lambda_), 1),
                                 round(max(self.lambda_), 1) + 0.1, 0.3)
            ax2.set_xticks(ax2Ticks)
            ax2.set_ylabel('$f_\\mathrm{s}$')
            ax2.set_xlabel('$\\lambda$')
            lines = ax2.get_lines()
            legend1 = plt.legend([lines[i] for i in [1, 3]],
                                 ['$f_\\mathrm{m}$', '$f_\\mathrm{f}$'], loc=1,
                                 fontsize='medium', mode='expand', ncol=2,
                                 framealpha=0.25)
            ax2.add_artist(legend1)
            if self.FS['lambda'] is not None:
                legend2 = plt.legend(
                    [lines[i] for i in [4]],
                    ['$f_\\mathrm{s}=' + str(round(self.FS['fs'], 3)) + '$' +
                     '\n$\\lambda=' + str(round(self.FS['lambda'], 3))+'$'],
                    framealpha=0.25, fontsize='medium', loc=8)
                ax2.add_artist(legend2)
        else:
            plt.legend(loc=2, framealpha=0.25, fontsize='medium')
        fig.tight_layout()
        return fig


# %%
'''
BSD 2 license.

Copyright (c) 2018, Universidad Nacional de Colombia, Exneyder Andres Montoya
    Araque and Ludger O. Suarez-Burgoa.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
