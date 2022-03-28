from .posetrackreid_api import PoseTrackReIDEvaluator 
from .posetrack_api import PoseTrackEvaluator 
from .posetrack_mot_api import PoseTrackMOTEvaluator 

__all__ = ['get_api',  'PoseTrackMOTEvaluator',  'PoseTrackEvaluator',  'PoseTrackReIDEvaluator']


def get_api(trackers_folder, gt_folder, use_parallel, num_parallel_cores, eval_type):
    """
    eval_type: ['pose_tracking', 'reid_tracking', 'posetrack_mot']
    """

    if eval_type == 'tracking':
        return PoseTrackEvaluator(trackers_folder, gt_folder, use_parallel, num_parallel_cores)
    elif eval_type == 'reid_tracking':
        return PoseTrackReIDEvaluator(trackers_folder, gt_folder)
    elif eval_type == 'posetrack_mot':
        return PoseTrackMOTEvaluator(trackers_folder, gt_folder, use_parallel, num_parallel_cores)
    else:
        raise NotImplementedError(f"No api for eval_type: {eval_type}")
