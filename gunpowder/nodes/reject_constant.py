import logging
import random
import numpy as np
from numpy.core.numeric import Inf

from .batch_filter import BatchFilter
from gunpowder.profiling import Timing
from .random_location import RandomLocation

logger = logging.getLogger(__name__)


class RejectConstant(BatchFilter):
    '''Reject batches deemed to be constant (and thus irrelevant likely due to imperfect data, e.g. missing slices) based on the coefficient of variation of the data.

    Args:

        array (:class:`ArrayKey`):

            The array to reject constant batches from.

        min_coefvar (``float``, optional):

            The minimal required coefficient of variation of the batch (i.e. standard deviation divided by mean).
            Defaults to 1e-04.

        reject_probability (``float``, optional):

            The probability by which a batch that is not valid is actually rejected. Defaults to 1., i.e. strict
            rejection.
            
    '''

    def __init__(
            self,
            array,
            min_coefvar=None,
            reject_probability=1.,
            axis=1 # should maybe be None
            ):

        self.array = array
        if min_coefvar is None:
            self.min_coefvar = 1e-04
        else:
            self.min_coefvar = min_coefvar
        self.reject_probability = reject_probability
        self.axis = axis

    def setup(self):
        assert self.array in self.spec, (
            "Reject can only be used if %s is provided" % self.array)
        self.upstream_provider = self.get_upstream_provider()

    def provide(self, request):
        if self.reject_probability < 1.:
            random.seed(request.random_seed)

        report_next_timeout = 10
        num_rejected = 0

        timing = Timing(self)
        timing.start()
        assert self.array in request, (
            "Reject can only be used if %s is provided" % self.array)

        have_good_batch = False
        first_seed = request.random_seed
        while not have_good_batch:

            batch = self.upstream_provider.request_batch(request)
            data = batch.arrays[self.array].data.squeeze()

            coefvar = abs(np.std(data, axis=self.axis)) / np.clip(abs(np.mean(data, axis=self.axis)), 1e-10, None) # ensure numerical stability if mean = 0

            have_good_batch = coefvar.min() > self.min_coefvar

            if not have_good_batch and self.reject_probability < 1.:
                have_good_batch = random.random() > self.reject_probability

            if not have_good_batch:
                logger.debug(
                    "reject batch with coefficient of variation %g at %s, first seed: %d",
                    coefvar.min(), batch.arrays[self.array].spec.roi, first_seed)
                num_rejected += 1

                if timing.elapsed() > report_next_timeout:

                    logger.warning(
                        "rejected %d batches, been waiting for a good one "
                        "since %ds", num_rejected, report_next_timeout)                        
                    logger.warning(
                        "last batch rejected with coefficient of variation %g at %s",
                        coefvar.min(), batch.arrays[self.array].spec.roi)
                    report_next_timeout *= 2

            else:
                logger.debug(
                    "accepted batch with coefficient of variation %g at %s",
                    coefvar.min(), batch.arrays[self.array].spec.roi)

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch
