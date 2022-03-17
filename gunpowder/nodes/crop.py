import logging

from .batch_filter import BatchFilter
from gunpowder.coordinate import Coordinate

logger = logging.getLogger(__name__)

class Crop(BatchFilter):
    '''Limits provided ROIs by either giving a new :class:`Roi` or crop
    fractions from either face of the provided ROI.

    Args:

        key (:class:`ArrayKey` or :class:`GraphKey`):

            The key of the array or points set to modify.

        roi (:class:`Roi` or ``None``):

            The ROI to crop to.

        fraction_negative (``tuple`` of ``float``):

            Relative crop starting from the negative end of the provided ROI.

        fraction_positive (``tuple`` of ``float``):

            Relative crop starting from the positive end of the provided ROI.

        abs_negative (``tuple`` of ``int`` or :class:`Coordinate`):

            Number of pixels to crop starting from the negative end of the provided ROI.

        abs_positive (``tuple`` of ``int`` or :class:`Coordinate`):

            Number of pixels to crop starting from the positive end of the provided ROI.
    '''

    def __init__(
            self,
            key,
            roi=None,
            fraction_negative=None,
            fraction_positive=None,
            abs_negative=None,
            abs_positive=None):

        if roi is not None and (
                fraction_positive is not None or
                fraction_negative is not None or
                abs_positive is not None or
                abs_negative is not None):
            raise RuntimeError(
                "'roi' and 'fraction_...' arguments can not be given together")
        if (fraction_positive is not None and abs_positive is not None) or (fraction_negative is not None  and abs_negative is not None):
            raise RuntimeError(
                "'fraction_...' and 'abs_...' arguments can not be given together")

        if all([var is None for var in (roi, fraction_positive, fraction_negative, abs_positive, abs_negative)]):
            raise RuntimeError(
                "One of 'roi', 'fraction_...', or 'abs_...' has to be given")

        if fraction_negative is not None and fraction_positive is None:
            fraction_positive = (0.0,)*len(fraction_negative)
        elif fraction_positive is not None and fraction_negative is None:
            fraction_negative = (0.0,)*len(fraction_positive)

        self.key = key
        self.roi = roi
        self.fraction_negative = fraction_negative
        self.fraction_positive = fraction_positive
        self.abs_negative = abs_negative
        self.abs_positive = abs_positive

    def setup(self):

        spec = self.spec[self.key]

        if self.roi is not None:

            assert spec.roi.contains(self.roi), (
                "Crop ROI is not contained in upstream ROI.")

            cropped_roi = self.roi

        else:

            if self.fraction_negative:
                total_fraction = tuple(
                    n + p
                    for n, p in zip(self.fraction_negative, self.fraction_positive)
                )
                if max(total_fraction) >= 1:
                    raise RuntimeError("Sum of crop fractions exeeds 1")
            
            if self.abs_positive is not None:
                crop_positive = Coordinate(self.abs_positive)
            else:
                crop_positive = spec.roi.get_shape()*self.fraction_positive

            if self.abs_negative is not None:
                crop_negative = Coordinate(self.abs_negative)
            else:
                crop_negative = spec.roi.get_shape()*self.fraction_negative
            
            if any([(crop_positive + crop_negative) >= spec.roi.get_shape()]):
                raise RuntimeError(f'Combination of requested positive crop ({crop_positive}) and requested negative crop ({crop_negative}) are larger than the provided ROI ({spec.roi.get_shape()}).')

            cropped_roi = spec.roi.grow(
                -crop_positive,
                -crop_negative)

        spec.roi = cropped_roi
        self.updates(self.key, spec)

    def process(self, batch, request):
        pass
