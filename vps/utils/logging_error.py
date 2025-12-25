import numpy as np
import logging
from .metric import get_translation_error, get_transl_ang_err, get_rot_err

def logging_error(model_pred_pose,final_pose,q_gt_pose,refs_gt_pose):
    """
    - model_pred: list [4x4] model pred (C2W) query+refs
    - final_pose: 4x4 fixpred query2world c2w
    - q_gt: 4x4 gt (C2W) query2world
    - refs_gt: list[4x4] gt (C2W) refs2world
    """
    
    logging.info("")
    logging.info("=== 初步Query结果 vs GT pose误差分析 ===")
    #查gt_pose
    # 计算平移误差（欧几里得距离）
    P_query = model_pred_pose[0]
    P_refs = model_pred_pose[1:]
    P_ref = P_refs[0]
    query2ref = np.linalg.inv(P_ref) @ P_query
    q_pred_pose = refs_gt_pose[0] @ query2ref #预测的query2world 相较于数据库坐标系

    t_final = q_pred_pose[:3, 3]
    t_gt = q_gt_pose[:3, 3]
    translation_error = get_translation_error(t_final, t_gt)
    
    # 计算位移方向夹角
    translation_angle_error = get_transl_ang_err(t_final, t_gt)
    # 计算scale：两个pose在各自位移方向上的距离比例
    t_final_magnitude = np.linalg.norm(t_final)  # final_pose的位移距离
    t_gt_magnitude = np.linalg.norm(t_gt)        # GT pose的位移距离
    scale_ratio = t_final_magnitude / t_gt_magnitude  # scale = m/n
    
    # 计算旋转误差（角度）
    R_final = q_pred_pose[:3, :3]
    R_gt = q_gt_pose[:3, :3]
    rotation_angle_error = get_rot_err(R_final, R_gt)
    
    logging.info(f"平移误差 = {translation_error:.4f}m, 位移方向夹角 = {translation_angle_error:.2f}°, scale比例 = {scale_ratio:.4f}, 旋转误差 = {rotation_angle_error:.2f}°")





#     q2r_pred_poses = [np.linalg.inv(P_ref) @ P_query for P_ref in model_pred_pose[1:]]
# # 计算GT的q2r相对位移和旋转
#     q2r_gt_poses = []   
#     for ref_gt_pose in refs_gt_pose:
#         # 计算GT的相对变换：从query到ref
#         q2r_gt = np.linalg.inv(ref_gt_pose) @ q_gt_pose
#         q2r_gt_poses.append(q2r_gt)
    
#     # 计算误差
#     logging.info("=== GT vs 估计值误差分析 ===")
#     for i, (q2r_est, q2r_gt) in enumerate(zip(q2r_pred_poses, q2r_gt_poses)):
#         # 平移误差（方向）
#         t_est = q2r_est[:3, 3] / np.linalg.norm(q2r_est[:3, 3])  # 归一化方向
#         t_gt = q2r_gt[:3, 3] / np.linalg.norm(q2r_gt[:3, 3])    # 归一化方向
        
#         # 角度误差（度）
#         translation_angle_error = get_transl_ang_err(t_est,t_gt)
#         translation_error = get_translation_error(t_est,t_gt)
        
#         # 旋转误差
#         R_est = q2r_est[:3, :3]
#         R_gt = q2r_gt[:3, :3]
#         rotation_angle_error = get_rot_err(R_est, R_gt)
#         logging.info(f"参考图像 {i}: 平移方向误差 = {translation_angle_error:.2f}°,平移差异 = {translation_error:.2f}, 旋转误差 = {rotation_angle_error:.2f}°")
#     logging.info("")
#     logging.info("================================================")
#     # 计算优化后的final_pose vs GT pose的误差分析
#     logging.info("=== 优化后的final_pose vs GT pose误差分析 ===")
#     # 计算final_pose相对于每个ref的相对变换
#     q2r_final_poses = []
#     for ref_gt_pose in refs_gt_pose:
#         # 计算final_pose的相对变换：从final_pose到ref
#         q2r_final = np.linalg.inv(ref_gt_pose) @ final_pose
#         q2r_final_poses.append(q2r_final)
    
#     # 计算误差：比较优化后的final_pose与GT pose
#     for i, (q2r_final, q2r_gt) in enumerate(zip(q2r_final_poses, q2r_gt_poses)):
#         # 平移误差（方向）
#         t_final = q2r_final[:3, 3] / np.linalg.norm(q2r_final[:3, 3])  # 归一化方向
#         t_gt = q2r_gt[:3, 3] / np.linalg.norm(q2r_gt[:3, 3])    # 归一化方向
        
#         # 角度误差（度）
#         translation_angle_error = get_transl_ang_err(t_final,t_gt)
#         translation_error = get_translation_error(t_final,t_gt)
        
#         # 旋转误差
#         R_final = q2r_final[:3, :3]
#         R_gt = q2r_gt[:3, :3]
#         rotation_angle_error = get_rot_err(R_final,R_gt)
        
#         logging.info(f"参考图像 {i}: 平移方向误差 = {translation_angle_error:.2f}°,平移差异 = {translation_error:.2f}, 旋转误差 = {rotation_angle_error:.2f}°")
    
    # 计算初步query结果与GT之间的误差
    logging.info("")
    logging.info("=== 优化后Query结果 vs GT pose误差分析 ===")
    # 计算平移误差（欧几里得距离）
    t_final = final_pose[:3, 3]
    t_gt = q_gt_pose[:3, 3]
    translation_error = get_translation_error(t_final,t_gt)
    
    # 计算位移方向夹角
    translation_angle_error = get_transl_ang_err(t_final,t_gt)
    
    # 计算scale：两个pose在各自位移方向上的距离比例
    t_final_magnitude = np.linalg.norm(t_final)  # final_pose的位移距离
    t_gt_magnitude = np.linalg.norm(t_gt)        # GT pose的位移距离
    scale_ratio = t_final_magnitude / t_gt_magnitude  # scale = m/n
    
    # 计算旋转误差（角度）
    R_final = final_pose[:3, :3]
    R_gt = q_gt_pose[:3, :3]
   
    rotation_angle_error = get_rot_err(R_final,R_gt)
    
    logging.info(f"平移误差 = {translation_error:.4f}m, 位移方向夹角 = {translation_angle_error:.2f}°, scale比例 = {scale_ratio:.4f}, 旋转误差 = {rotation_angle_error:.2f}°")
