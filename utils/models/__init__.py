import sys

from utils.models.mlp_model import *
from utils.models.lenet1 import *
from utils.models.lenet5 import *
from utils.models.vgg16 import *
from utils.models.resnet20 import *
from utils.models.imdb_transformer import IMDB_Transformer
from utils.models.sms_bilstm import SMS_BiLSTM

from utils.models.fm_simplenet import FM_SimpleNet
# import utils.models.fm_simplenet as fm_simple_Net_withForword2
# sys.modules["lhr.lhr_utils.models.fm_simple_Net_withForword2"] = fm_simple_Net_withForword2

from utils.models.c10_simplenet import C10_SimpleNet
from utils.models.c10_cnn import C10_CNN
# import utils.models.c10_cnn as C10_CNN1_Net_withForword2_v2
# sys.modules["lhr.lhr_utils.models.c10_models"] = C10_CNN1_Net_withForword2_v2

from utils.models.guide_fcn import GUIDE_FCN
from utils.models.usairline_lstm import USAirline_LSTM

# Import for deserialization of fastvit

from utils.models.ml_fastvit.models import FastViT_T8, FastViT_S12
import utils.models.ml_fastvit as ml_fastvit
sys.modules["lhr"] = ml_fastvit
sys.modules["lhr.lhr_utils.models.ml_fastvit"] = ml_fastvit
