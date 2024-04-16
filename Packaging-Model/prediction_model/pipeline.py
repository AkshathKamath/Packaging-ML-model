from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import prediction_model.processing.pre_processing as pp
from prediction_model.config import config

classification_pipeline = Pipeline(
    [
        ('MedianImputation', pp.MedianImputer(variables = config.NUM_FEATURES)),
        ('ModeImputation',pp.ModeImputer(variables = config.CAT_FEATURES)),
        ('AddColumns', pp.AddColumns(col1 = config.FEATURE_TO_MODIFY, col2 = config.FEATURE_TO_ADD )),
        ('DropFeatures', pp.DropColumns(variables = config.DROP_FEATURES)),
        ('LabelEncoder',pp.LabelEncoder(variables = config.FEATURES_TO_ENCODE)),
        ('LogTransform',pp.LogTransformer(variables = config.LOG_FEATURES)),
        ('MinMaxScale', MinMaxScaler()),
        ('LogisticClassifier',LogisticRegression(random_state = 0))

    ]
)