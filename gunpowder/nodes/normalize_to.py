import logging
import copy
import numpy as np

from gunpowder.batch_request import BatchRequest

from .batch_filter import BatchFilter

logger = logging.getLogger(__name__)

class NormalizeTo(BatchFilter):
    '''Normalize the values of an array to be floats on a given interval.

    Args:

        array (:class:`ArrayKey`):

            The key of the array to modify.

        out_interval (tuple, optional):

            The interval to normalize inputs to. Defaults to [0, 1]. However, [-1, 1] would be helpful for inputs to networks using tanh as an activation function, for example.

        in_interval (tuple, optional):

            The interval of inputs to normalize from. If not given, a factor is chosen based on the
            ``dtype`` of the array (e.g., ``np.uint8`` would result in a factor
            of ``1.0/255``).

        dtype (data-type, optional):

            The datatype of the normalized array. Defaults to ``np.float32``.
    '''

    def __init__(self, array, out_interval=[0,1], in_interval=None, dtype=np.float32):

        self.array = array
        self.out_interval = out_interval
        self.out_factor = 1 / abs(np.diff(self.out_interval)[0])    # determine intended width of output distribution
        self.out_offset = np.mean(self.out_interval)                # determine intended center of output distribution
        self.in_interval = in_interval
        if self.in_interval is not None:
            self.in_factor = 1 / abs(np.diff(self.in_interval)[0])  # determine expected width of input distribution
            self.in_offset = np.mean(self.in_interval)              # determine expected center of input distribution

        else:
            self.factor = None
        self.dtype = dtype

    def setup(self):
        self.enable_autoskip()
        array_spec = copy.deepcopy(self.spec[self.array])
        array_spec.dtype = self.dtype
        self.updates(self.array, array_spec)

    def prepare(self, request):
        deps = BatchRequest()
        deps[self.array] = request[self.array]
        deps[self.array].dtype = None
        return deps

    def process(self, batch, request):

        if self.array not in batch.arrays:
            return

        array = batch.arrays[self.array]
        array.spec.dtype = self.dtype

        factor = self.factor            
        if factor is None:

            logger.debug("automatically normalizing %s with dtype=%s",
                    self.array, array.data.dtype)

            if array.data.dtype == np.uint8:
                factor = 1.0/255
            elif array.data.dtype == np.uint16:
                factor = 1.0/(256*256-1)
            elif array.data.dtype == np.float32:
                if array.data.min() < 0 and array.data.min() >= -1 and array.data.max() <= 1: # Assume follows tanh activation function (i.e. on [-1, 1])
                    logger.debug('Normalization assumes data is produced by tanh activation function or similar (i.e. on [-1, 1]).')
                    array.data += 1
                    array.data /= 2
                assert array.data.min() >= 0 and array.data.max() <= 1, (
                        "Values are float but not in [0,1], I don't know how "
                        "to normalize. Please provide a factor.")
                factor = 1.0
            else:
                raise RuntimeError("Automatic normalization for " +
                        str(array.data.dtype) + " not implemented, please "
                        "provide a factor.")

        logger.debug("scaling %s with %f", self.array, factor)
        array.data = array.data.astype(self.dtype)*factor
