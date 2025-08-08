# # /share/project/zhipengchen/dj/verl-dapo-smo/verl/trainer/ppo/load_matrix.py

# from scipy.sparse import load_npz

# # 将 compressed_matrix 定义为全局变量
# compressed_matrix = None
# def load_similarity_matrix():  # 使用全局变量
#     # 只有在 compressed_matrix 为 None 时才加载矩阵
#     global compressed_matrix
#     if compressed_matrix is None:
#         # 加载保存的压缩稀疏矩阵
#         compressed_matrix = load_npz("/share/project/zhipengchen/dj/RFT/code/compressed_selected_token_similarity.npz")
#         print("Similarity matrix loaded.")
#     return compressed_matrix
# # 自动加载矩阵，在模块导入时调用 load_similarity_matrix
# load_similarity_matrix()

import torch

token_math_flag_tensor = None


def load_token_math_flag_tensor():
    global token_math_flag_tensor
    if token_math_flag_tensor is None:
        token_math_flag_tensor = torch.load(
            "/opt/aps/workdir/jiechen/train/token_math_mask/token_math_flag.pt"
        )
        print("================\nToken Math Flag Tensor loaded.\n================")
    return token_math_flag_tensor


load_token_math_flag_tensor()
