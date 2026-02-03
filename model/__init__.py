from .I3D.pytorch_i3d import InceptionI3d
from .Spatial_Temporal.ViolenceDetector import ViolenceDetectionModel, build_violence_detection_model
from .ResNet3D.model import generate_model as generate_resnet3d_model
from .TSM.ops.models import TSN
from .TDN.ops.tdn_net import tdn_net
from .TimeSformer.model import TimeSformerModel, get_timesformer_model
from .X3D.X3D import X3DModel, get_model_x3d
# from .TCM.TCM import TCM

def get_model(model_name, **kwargs):
    """
    Factory function to get model by name with configuration.
    
    Args:
        model_name (str): Name of the model
        **kwargs: Model-specific configuration parameters
        
    Returns:
        model: Instantiated model or model class
    """
    pretrained_weights = kwargs.get('pretrained_weights', None)

    if model_name == "I3D":
        num_classes = kwargs.get('num_classes', 400)
        in_channels = kwargs.get('in_channels', 3)
        dropout_keep_prob = kwargs.get('dropout_keep_prob', 0.5)
        return InceptionI3d(
            num_classes=num_classes,
            in_channels=in_channels,
            dropout_keep_prob=dropout_keep_prob
        )
    
    elif model_name == "ViolenceDetection" or model_name == "LSTM_CNN":
        # Use the build function from ViolenceDetector
        return build_violence_detection_model(**kwargs)
    
    elif model_name == "ResNet3D":
        # Return the generate_model function that needs options
        return generate_resnet3d_model
    
    elif model_name == "TSM":
        num_class = kwargs.get('num_class', 400)
        num_segments = kwargs.get('num_segments', 8)
        modality = kwargs.get('modality', 'RGB')
        base_model = kwargs.get('base_model', 'resnet50')
        consensus_type = kwargs.get('consensus_type', 'avg')
        dropout = kwargs.get('dropout', 0.5)
        
        return TSN(
            num_class=num_class,
            num_segments=num_segments,
            modality=modality,
            base_model=base_model,
            consensus_type=consensus_type,
            dropout=dropout
        )
    
    elif model_name == "TDN":
        base_model = kwargs.get('base_model', 'resnet50')
        num_segments = kwargs.get('num_segments', 8)
        pretrained = kwargs.get('pretrained', True)
        
        return tdn_net(
            base_model=base_model,
            num_segments=num_segments,
            pretrained=pretrained
        )
    
    elif model_name == 'TimeSformer':
        return get_timesformer_model(**kwargs)
    
    elif model_name == 'X3D':
        return get_model_x3d(**kwargs)
    
    # elif model_name == "TCM":
    #     num_segments = kwargs.get('num_segments', 8)
    #     expansion = kwargs.get('expansion', 1)
    #     pos = kwargs.get('pos', 2)
        
    #     return TCM(
    #         num_segments=num_segments,
    #         expansion=expansion,
    #         pos=pos
    #     )
    
    else:
        raise ValueError(f"Model {model_name} not recognized. Available models: I3D, ViolenceDetection, LSTM_CNN, ResNet3D, TSN, TDN, TimeSformer, X3D")