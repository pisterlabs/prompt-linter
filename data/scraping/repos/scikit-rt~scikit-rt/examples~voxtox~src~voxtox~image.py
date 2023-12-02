'''VoxTox extenstions to Scikit-rt for Images'''

import skrt

from voxtox.core import get_couch_shifts

class Image(skrt.image.Image):
    '''VoxTox-specific extensions to Scikit-rt Image class.'''

    def apply_couch_shifts(self, reverse=False, order=0):
        '''
        Apply couch shifts.

        Couch shifts - applied in the order translation, rotation - represent
        the transformation for mapping from guidance scan to planning scan.

        Reverse shifts - applied in the order rotation, translation - represent
        the transformation for mapping from planning scan to guidance scan.

        **Parameters:**

        reverse: bool, default=False
            If True, reverse couch shifts are applied.

        order: int, default = 1
            Order of the b-spline used in interpolating voxel intensity values.
        '''
        translation, rotation = get_couch_shifts(self, reverse)
        if reverse:
            self.transform(rotation=rotation, order=order)
            self.transform(translation=translation, order=order)
        else:
            self.transform(translation=translation, order=order)
            self.transform(rotation=rotation, order=order)
