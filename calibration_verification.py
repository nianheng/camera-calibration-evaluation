# -*- coding: utf-8 -*-

import math
import numpy as np

from dual_quaternion_hand_eye_calibration import (
    evaluate_alignment, align_paths_at_index, get_aligned_poses, compute_pose_error)
from dual_quaternion import DualQuaternion
from hand_eye_calibration_plotting_tools import plot_poses


def evaluate_calibration(aligned_poses_B_H, aligned_poses_W_E, poses_H_E):
    
    dq_H_E = DualQuaternion.from_pose_vector(poses_H_E)
    dq_H_E.normalize()
    
    # Convert poses to dual quaterions.
    dual_quat_B_H_vec = [DualQuaternion.from_pose_vector(
        aligned_pose_B_H) for aligned_pose_B_H in aligned_poses_B_H]
    dual_quat_W_E_vec = [DualQuaternion.from_pose_vector(
        aligned_pose_W_E).inverse() for aligned_pose_W_E in aligned_poses_W_E]

    # aligned_dq_B_H = dual_quat_B_H_vec
    # aligned_dq_W_E = dual_quat_W_E_vec

    aligned_dq_B_H = align_paths_at_index(dual_quat_B_H_vec, 0)
    aligned_dq_W_E = align_paths_at_index(dual_quat_W_E_vec, 0)

    (poses_B_H, poses_W_H) = get_aligned_poses(aligned_dq_B_H,
                                                aligned_dq_W_E,
                                                dq_H_E)

    (rmse_position,
    rmse_orientation) = evaluate_alignment(poses_B_H, poses_W_H)

    # visualize:
    # every_nth_element = 1
    # plot_poses([poses_B_H[:: every_nth_element],
    #             poses_W_H[:: every_nth_element]],
    #             True, title="3D Poses After Fine Alignment")

    return (rmse_position, rmse_orientation)


# def compute_loop_error(results_dq_H_E, num_poses_sets, visualize=False):
#   calibration_transformation_chain = []

#   # Add point at origin to represent the first coordinate
#   # frame in the chain of transformations.
#   calibration_transformation_chain.append(
#       DualQuaternion(Quaternion(0, 0, 0, 1), Quaternion(0, 0, 0, 0)))

#   # Add first transformation
#   calibration_transformation_chain.append(results_dq_H_E[0])

#   # Create chain of transformations from the first frame to the last.
#   i = 0
#   idx = 0
#   while i < (num_poses_sets - 2):
#     idx += (num_poses_sets - i - 1)
#     calibration_transformation_chain.append(results_dq_H_E[idx])
#     i += 1

#   # Add inverse of first to last frame to close the loop.
#   calibration_transformation_chain.append(
#       results_dq_H_E[num_poses_sets - 2].inverse())

#   # Check loop.
#   assert len(calibration_transformation_chain) == (num_poses_sets + 1), (
#       len(calibration_transformation_chain), (num_poses_sets + 1))

#   # Chain the transformations together to get points we can plot.
#   poses_to_plot = []
#   dq_tmp = DualQuaternion(Quaternion(0, 0, 0, 1), Quaternion(0, 0, 0, 0))
#   for i in range(0, len(calibration_transformation_chain)):
#     dq_tmp *= calibration_transformation_chain[i]
#     poses_to_plot.append(dq_tmp.to_pose())

#   (loop_error_position, loop_error_orientation) = compute_pose_error(poses_to_plot[0],
#                                                                      poses_to_plot[-1])

#   print("Error when closing the loop of hand eye calibrations - position: {}"
#         " m orientation: {} deg".format(loop_error_position,
#                                         loop_error_orientation))

#   if visualize:
#     assert len(poses_to_plot) == len(calibration_transformation_chain)
#     plot_poses([np.array(poses_to_plot)], plot_arrows=True,
#                title="Hand-Eye Calibration Results - Closing The Loop")

#   # Compute error of loop.
#   return (loop_error_position, loop_error_orientation)
