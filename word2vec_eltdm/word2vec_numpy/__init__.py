from .models import SimpleWord2Vec, NegWord2Vec
from .optimizers import Optimizer, OptimizeNSL
from .losses import CrossEntropy, NegativeSamplingLoss
from .utils import train_default, train_NSL, update_best_loss
