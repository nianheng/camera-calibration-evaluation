# -*- coding: utf-8 -*-

from itertools import compress
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

import copy
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
# import tf
import timeit

from dual_quaternion import DualQuaternion
from quaternion import (
    Quaternion, angle_between_quaternions)
from hand_eye_calibration_plotting_tools import (
    plot_alignment_errors, plot_poses)

# This implements the following paper.
#
# @article{doi:10.1177/02783649922066213,
# author = {Konstantinos Daniilidis},
# title = {Hand-Eye Calibration Using Dual Quaternions},
# journal = {The International Journal of Robotics Research},
# volume = {18},
# number = {3},
# pages = {286-298},
# year = {1999},
# doi = {10.1177/02783649922066213},
# URL = {http://dx.doi.org/10.1177/02783649922066213},
# eprint = {http://dx.doi.org/10.1177/02783649922066213},
# }

# All Quaternions are Hamiltonian Quaternions.
# Denoted as: q = [x, y, z, w]

# Notations:
# Frames are:
# H: Hand frame
# B: World (Base) frame of hand
# E: Eye frame
# W: World frame of eye
#
# T_B_W: Denotes the transformation from a point in the World frame to the
# base frame.


class HandEyeConfig:

  def __init__(self):

    # General config.
    self.algorithm_name = ""
    self.use_baseline_approach = False
    self.min_num_inliers = 10
    self.enable_exhaustive_search = False

    # Select distinctive poses based on skrew axis
    self.prefilter_poses_enabled = True
    self.prefilter_dot_product_threshold = 0.975

    # Hand-calibration
    self.hand_eye_calibration_scalar_part_equality_tolerance = 4e-2

    # Visualization
    self.visualize = False
    self.visualize_plot_every_nth_pose = 10


def compute_dual_quaternions_with_offset(dq_B_H_vec, dq_H_E, dq_B_W):
  n_samples = len(dq_B_H_vec)
  dq_W_E_vec = []

  dq_W_B = dq_B_W.inverse()
  for i in range(0, n_samples):
    dq_B_H = dq_B_H_vec[i]

    dq_W_E = dq_W_B * dq_B_H * dq_H_E

    dq_W_E.normalize()
    assert np.isclose(dq_W_E.norm()[0], 1.0, atol=1.e-8), dq_W_E
    dq_W_E_vec.append(dq_W_E)
  return dq_W_E_vec


def align_paths_at_index(dq_vec, align_index=0, enforce_positive_q_rot_w=True):
  dq_align_inverse = dq_vec[align_index].inverse().copy()
  n_samples = len(dq_vec)
  dq_vec_starting_at_origin = [None] * n_samples
  for i in range(0, n_samples):
    dq_vec_starting_at_origin[i] = dq_align_inverse * dq_vec[i].copy()
    if (enforce_positive_q_rot_w):
      if dq_vec_starting_at_origin[i].q_rot.w < 0.:
        dq_vec_starting_at_origin[i].dq = -(
            dq_vec_starting_at_origin[i].dq.copy())

  # Rearange poses such that it starts at the origin.
  dq_vec_rearanged = dq_vec_starting_at_origin[align_index:] + \
      dq_vec_starting_at_origin[:align_index]


  return dq_vec_rearanged


def skew_from_vector(vector):
  skew = np.array([[0, -vector[2], vector[1]], [vector[2], 0, -vector[0]],
                   [-vector[1], vector[0], 0]])
  return skew.copy()


def setup_s_matrix(dq_1, dq_2):
  """This sets up the [6x8] S matrix, see Eq. (31) of the referenced paper.

  S = (skew(I(qr1)+I(qr2)) I(qr1)-I(qr2) 0_{3x3}             0_{3x1}      )
      (skew(I(qt1)+I(qt2)) I(qt1)-I(qt2) skew(I(qr1)+I(qr2)) I(qr1)-I(qr2))
  I(q) denotes the vector of the imaginary components of a quaternion.
  Note: The order of the blocks switched as we are using q = [x y z w]^T
  instead of q = [w x y z].T.
  """
  scalar_parts_1 = dq_1.scalar()
  scalar_parts_2 = dq_2.scalar()

  assert np.allclose(
      scalar_parts_1.dq, scalar_parts_2.dq,
      atol=5e-2), (
      "\ndq1:\n{},\nscalar_parts_1:\n{},\ndq2:\n{},\nscalar_parts_2:\n{}\n"
      "Scalar parts should always be equal.".format(dq_1, scalar_parts_1, dq_2,
                                                    scalar_parts_2))

  s_matrix = np.zeros([6, 8])
  s_matrix[0:3, 0:3] = skew_from_vector(dq_1.q_rot.q[0:-1] + dq_2.q_rot.q[0:-1])
  s_matrix[0:3, 3] = dq_1.q_rot.q[0:-1] - dq_2.q_rot.q[0:-1]
  s_matrix[3:6, 0:3] = skew_from_vector(dq_1.q_dual.q[0:-1] +
                                        dq_2.q_dual.q[0:-1])
  s_matrix[3:6, 3] = dq_1.q_dual.q[0:-1] - dq_2.q_dual.q[0:-1]
  s_matrix[3:6, 4:7] = skew_from_vector(dq_1.q_rot.q[0:-1] + dq_2.q_rot.q[0:-1])
  s_matrix[3:6, 7] = dq_1.q_rot.q[0:-1] - dq_2.q_rot.q[0:-1]
  # print("S: \n{}".format(s_matrix))

  rank_s_matrix = np.linalg.matrix_rank(s_matrix)
  assert rank_s_matrix <= 6, s_matrix
  return s_matrix.copy()


def setup_t_matrix(dq_W_E_vec, dq_B_H_vec):
  """This sets up the [6nx8] T matrix consisting of multiple S matrices for the
  different pose pairs. See Equation (33) of the referenced paper.

  T = (S_1.T S_2.T ... S_n.T).T
  """
  n_quaternions = len(dq_W_E_vec)
  t_matrix = np.zeros([6 * n_quaternions, 8])
  for i in range(n_quaternions):
    t_matrix[i * 6:i * 6 + 6, :] = setup_s_matrix(dq_W_E_vec[i], dq_B_H_vec[i])

  rank_t_matrix = np.linalg.matrix_rank(t_matrix, tol=5e-2)
  U, s, V = np.linalg.svd(t_matrix)
  # print("t_matrix: \n{}".format(t_matrix))
  # print("Rank(t_matrix): {}".format(rank_t_matrix))
  # assert rank_t_matrix == 6, ("T should have rank 6 otherwise we can not find "
  #                             "a rigid transform.", rank_t_matrix, s)
  return t_matrix.copy()


def compute_hand_eye_calibration(dq_B_H_vec_inliers, dq_W_E_vec_inliers,
                                 scalar_part_tolerance=1e-2,
                                 enforce_same_non_dual_scalar_sign=True):
  """
  Do the actual hand eye-calibration as described in the referenced paper.
  Assumes the outliers have already been removed and the scalar parts of
  each pair are a match.
  """
  n_quaternions = len(dq_B_H_vec_inliers)

  # Verify that the first pose is at the origin.
  assert np.allclose(dq_B_H_vec_inliers[0].dq,
                     [0., 0., 0., 1.0, 0., 0., 0., 0.],
                     atol=1.e-8), dq_B_H_vec_inliers[0]
  assert np.allclose(dq_W_E_vec_inliers[0].dq,
                     [0., 0., 0., 1.0, 0., 0., 0., 0.],
                     atol=1.e-8), dq_W_E_vec_inliers[0]

  if enforce_same_non_dual_scalar_sign:
    for i in range(n_quaternions):
      dq_W_E = dq_W_E_vec_inliers[i]
      dq_B_H = dq_B_H_vec_inliers[i]
      if ((dq_W_E.q_rot.w < 0. and dq_B_H.q_rot.w > 0.) or
              (dq_W_E.q_rot.w > 0. and dq_B_H.q_rot.w < 0.)):
        dq_W_E_vec_inliers[i].dq = -dq_W_E_vec_inliers[i].dq.copy()

  # 0. Stop alignment if there are still pairs that do not have matching
  # scalar parts.
  for j in range(n_quaternions):
    dq_B_H = dq_W_E_vec_inliers[j]
    dq_W_E = dq_B_H_vec_inliers[j]

    scalar_parts_B_H = dq_B_H.scalar()
    scalar_parts_W_E = dq_W_E.scalar()

    assert np.allclose(scalar_parts_B_H.dq, scalar_parts_W_E.dq,
                       atol=scalar_part_tolerance), (
        "Mismatch of scalar parts of dual quaternion at idx {}:"
        " dq_B_H: {} dq_W_E: {}".format(j, dq_B_H, dq_W_E))

  # 1.
  # Construct 6n x 8 matrix T
  t_matrix = setup_t_matrix(dq_B_H_vec_inliers, dq_W_E_vec_inliers)

  # 2.
  # Compute SVD of T and check if only two singular values are almost equal to
  # zero. Take the corresponding right-singular vectors (v_7 and v_8)
  U, s, V = np.linalg.svd(t_matrix)

  # Check if only the last two singular values are almost zero.
  bad_singular_values = False
  for i, singular_value in enumerate(s):
    if i < 6:
      if singular_value < 5e-1:
        bad_singular_values = True
    else:
      if singular_value > 5e-1:
        bad_singular_values = True
  v_7 = V[6, :].copy()
  v_8 = V[7, :].copy()
  # print("v_7: {}".format(v_7))
  # print("v_8: {}".format(v_8))

  # 3.
  # Compute the coefficients of (35) and solve it, finding two solutions for s.
  u_1 = v_7[0:4].copy()
  u_2 = v_8[0:4].copy()
  v_1 = v_7[4:8].copy()
  v_2 = v_8[4:8].copy()
  # print("u_1: {}, \nu_2: {}, \nv_1: {}, \nv_2: {}".format(u_1, u_2, v_1, v_2))

  a = np.dot(u_1.T, v_1)
  assert a != 0.0, "This would involve division by zero."
  b = np.dot(u_1.T, v_2) + np.dot(u_2.T, v_1)
  c = np.dot(u_2.T, v_2)
  # print("a: {}, b: {}, c: {}".format(a, b, c))
  square_root_term = b * b - 4.0 * a * c

  if square_root_term < -1e-2:
    assert False, "square_root_term is too negative: {}".format(
        square_root_term)
  if square_root_term < 0.0:
    square_root_term = 0.0
  s_1 = (-b + np.sqrt(square_root_term)) / (2.0 * a)
  s_2 = (-b - np.sqrt(square_root_term)) / (2.0 * a)
  # print("s_1: {}, s_2: {}".format(s_1, s_2))

  # 4.
  # For these two s values, compute s^2*u_1^T*u_1 + 2*s*u_1^T*u_2 + u_2^T*u_2
  # From these choose the largest to compute lambda_2 and then lambda_1
  solution_1 = s_1 * s_1 * np.dot(u_1.T, u_1) + 2.0 * \
      s_1 * np.dot(u_1.T, u_2) + np.dot(u_2.T, u_2)
  solution_2 = s_2 * s_2 * np.dot(u_1.T, u_1) + 2.0 * \
      s_2 * np.dot(u_1.T, u_2) + np.dot(u_2.T, u_2)

  if solution_1 > solution_2:
    assert solution_1 > 0.0, solution_1
    lambda_2 = np.sqrt(1.0 / solution_1)
    lambda_1 = s_1 * lambda_2
  else:
    assert solution_2 > 0.0, solution_2
    lambda_2 = np.sqrt(1.0 / solution_2)
    lambda_1 = s_2 * lambda_2
  # print("lambda_1: {}, lambda_2: {}".format(lambda_1, lambda_2))

  # 5.
  # The result is lambda_1*v_7 + lambda_2*v_8
  dq_H_E = DualQuaternion.from_vector(lambda_1 * v_7 + lambda_2 * v_8)
  # Normalize the output, to get rid of numerical errors.
  dq_H_E.normalize()

  if (dq_H_E.q_rot.w < 0.):
    dq_H_E.dq = -dq_H_E.dq.copy()
  return (dq_H_E, s, bad_singular_values)


def prefilter_using_screw_axis(dq_W_E_vec_in, dq_B_H_vec_in, dot_product_threshold=0.95):
  dq_W_E_vec = copy.deepcopy(dq_W_E_vec_in)
  dq_B_H_vec = copy.deepcopy(dq_B_H_vec_in)
  n_quaternions = len(dq_W_E_vec)
  i = 0
  while i < len(dq_W_E_vec):
    dq_W_E_i = dq_W_E_vec[i]
    dq_B_H_i = dq_B_H_vec[i]
    screw_axis_W_E_i, rotation_W_E_i, translation_W_E_i = dq_W_E_i.screw_axis()
    screw_axis_B_H_i, rotation_B_H_i, translation_B_H_i = dq_B_H_i.screw_axis()

    if (np.linalg.norm(screw_axis_W_E_i) <= 1.e-12 or np.linalg.norm(screw_axis_B_H_i) <= 1.e-12):
      dq_W_E_vec.pop(i)
      dq_B_H_vec.pop(i)
    else:
      screw_axis_W_E_i = screw_axis_W_E_i / np.linalg.norm(screw_axis_W_E_i)
      screw_axis_B_H_i = screw_axis_B_H_i / np.linalg.norm(screw_axis_B_H_i)

      # TODO(ntonci): Add a check for small motion

      j = i + 1
      while j < len(dq_W_E_vec):
        dq_W_E_j = dq_W_E_vec[j]
        dq_B_H_j = dq_B_H_vec[j]
        screw_axis_W_E_j, rotation_W_E_j, translation_W_E_j = dq_W_E_j.screw_axis()
        screw_axis_B_H_j, rotation_B_H_j, translation_B_H_j = dq_B_H_j.screw_axis()

        if (np.linalg.norm(screw_axis_W_E_j) <= 1.e-12 or np.linalg.norm(screw_axis_B_H_j) <= 1.e-12):
          dq_W_E_vec.pop(j)
          dq_B_H_vec.pop(j)
        else:
          screw_axis_W_E_j = screw_axis_W_E_j / np.linalg.norm(screw_axis_W_E_j)
          screw_axis_B_H_j = screw_axis_B_H_j / np.linalg.norm(screw_axis_B_H_j)

          if (np.inner(screw_axis_W_E_i, screw_axis_W_E_j) > dot_product_threshold):
            dq_W_E_vec.pop(j)
            dq_B_H_vec.pop(j)
          elif (np.inner(screw_axis_B_H_i, screw_axis_B_H_j) > dot_product_threshold):
            dq_W_E_vec.pop(j)
            dq_B_H_vec.pop(j)
          else:
            j += 1
      i += 1

  assert i >= 2, "Not enough distinct poses found."
  return dq_W_E_vec, dq_B_H_vec


def compute_pose_error(pose_A, pose_B):
  """
  Compute the error norm of position and orientation.
  """
#   error_position = np.linalg.norm(pose_A[0:3] - pose_B[0:3], ord=2)
  error_position =  pose_A[0:3] - pose_B[0:3]

  # Construct quaternions to compare.
  quaternion_A = Quaternion(q=pose_A[3:7])
  quaternion_A.normalize()
  if quaternion_A.w < 0:
    quaternion_A.q = -quaternion_A.q
  quaternion_B = Quaternion(q=pose_B[3:7])
  quaternion_B.normalize()
  if quaternion_B.w < 0:
    quaternion_B.q = -quaternion_B.q

  # Sum up the square of the orientation angle error.
  error_angle_rad = angle_between_quaternions(
      quaternion_A, quaternion_B)
  error_angle_degrees = math.degrees(error_angle_rad)
  if error_angle_degrees > 180.0:
    error_angle_degrees = math.fabs(360.0 - error_angle_degrees)

  return (error_position, error_angle_degrees)


def evaluate_alignment(poses_A, poses_B, visualize=True):
  """
  Takes aligned poses and compares position and orientation.
  Returns the RMSE of position and orientation as well as a bool vector,
  indicating which pairs are below the error thresholds specified in the
  configuration:
    ransac_orientation_error_threshold_deg
    ransac_position_error_threshold_m
  """

  assert np.array_equal(poses_A.shape, poses_B.shape), (
      "Two pose vector of different size cannot be evaluated. "
      "Size pose A: {} Size pose B: {}".format(poses_A.shape, poses_B.shape))
  assert poses_A.shape[1] == 7, "poses_A are not valid poses!"
  assert poses_B.shape[1] == 7, "poses_B are not valid poses!"

  num_poses = poses_A.shape[0]

#   inlier_list = [False] * num_poses
  errors_position = np.zeros((num_poses, 3))
  errors_orientation = np.zeros((num_poses, 1))

#   errors_position = np.zeros((num_poses, 1))
#   errors_orientation = np.zeros((num_poses, 1))
  for i in range(0, num_poses):
    (error_position,
     error_angle_degrees) = compute_pose_error(poses_A[i, :], poses_B[i, :])

    # if (error_angle_degrees < config.ransac_orientation_error_threshold_deg and
    #         error_position < config.ransac_position_error_threshold_m):
    #   inlier_list[i] = True

    errors_position[i] = error_position
    errors_orientation[i] = error_angle_degrees

#   rmse_pose_accumulator = np.sum(np.square(errors_position))
#   rmse_orientation_accumulator = np.sum(np.square(errors_orientation))

  error_position_avg = np.mean(errors_position, axis=0)
  errors_orientation_avg = np.mean(errors_orientation)

#   rmse_pose_accumulator = 0
#   rmse_orientation_accumulator = 0
  rmse_pose_accumulator = np.array([0, 0, 0])
  rmse_orientation_accumulator = 0

  for i in range(0, num_poses):
    rmse_pose_accumulator = rmse_pose_accumulator + \
        np.square(errors_position[i] - error_position_avg)
    rmse_orientation_accumulator = rmse_orientation_accumulator + \
        np.square(errors_orientation[i] - errors_orientation_avg)

  # Compute RMSE.
  rmse_pose = np.zeros(3)
  for i in range(0, 3):
    rmse_pose[i] = math.sqrt(rmse_pose_accumulator[i] / num_poses)
#   rmse_pose = np.array([math.sqrt(rmse_pose_accumulator[i] / num_poses)] for i in range(0, 3))
  rmse_orientation = math.sqrt(rmse_orientation_accumulator / num_poses)

#   Plot the error.
#   if visualize:
#     plot_alignment_errors(errors_position, rmse_pose, errors_orientation,
#                           rmse_orientation, blocking=True)

  return (rmse_pose, rmse_orientation)


def get_aligned_poses(dq_B_H_vec, dq_W_E_vec, dq_H_E_estimated):
  
  # Compute aligned poses.
  dq_E_H_estimated = dq_H_E_estimated.inverse()
  dq_E_H_estimated.normalize()
  dq_E_H_estimated.enforce_positive_q_rot_w()

  dq_W_H_vec = []
  for i in range(0, len(dq_B_H_vec)):
    dq_W_H = dq_W_E_vec[i] * dq_E_H_estimated
    dq_W_H.normalize()

    if ((dq_W_H.q_rot.w < 0. and dq_B_H_vec[i].q_rot.w > 0.) or
            (dq_W_H.q_rot.w > 0. and dq_B_H_vec[i].q_rot.w < 0.)):
      dq_W_H.dq = -dq_W_H.dq.copy()

    dq_W_H_vec.append(dq_W_H)

  dq_W_H_vec = align_paths_at_index(dq_W_H_vec)

  # Convert to poses.
  poses_W_H = np.array([dq_W_H_vec[0].to_pose().T])
  for i in range(1, len(dq_W_H_vec)):
    poses_W_H = np.append(poses_W_H, np.array(
        [dq_W_H_vec[i].to_pose().T]), axis=0)
  poses_B_H = np.array([dq_B_H_vec[0].to_pose().T])
  for i in range(1, len(dq_B_H_vec)):
    poses_B_H = np.append(poses_B_H, np.array(
        [dq_B_H_vec[i].to_pose().T]), axis=0)

  return (poses_B_H.copy(), poses_W_H.copy())

