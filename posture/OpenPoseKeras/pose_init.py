
import cv2
import numpy as np

pose_model = None
pose_params = None
pose_model_params = None

def pose_init(weights):
    """
    Load Pose Model only ONCE.
    Call this BEFORE using Dataset / Dataloader loop.
    """
    global pose_model, pose_params, pose_model_params
    
    if pose_model is not None:
        print("Pose model already initialized ✅")
        return

    print("Loading Pose model (one-time init)...")

    # Import actual functions
    from model import get_testing_model
    from config_reader_colab import config_reader_colab

    # Build & load model
    pose_model = get_testing_model(np_branch1=38, np_branch2=19, stages=6)
    pose_model.load_weights(weights)

    # Load config params
    pose_params, pose_model_params = config_reader_colab()

    print("Pose model loaded ✅")


def pose_process(img):
    """
    Run pose process on an already-loaded image (numpy BGR or RGB)
    """
    global pose_model, pose_params, pose_model_params
    
    if pose_model is None:
        raise RuntimeError("❌ pose_model not initialized! Call pose_init(...) first.")

    from demo_image import process 
    canvas = process(img, pose_params, pose_model_params,pose_model)
    return canvas
