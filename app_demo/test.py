'''
Author: tany 1767774755@qq.com
Date: 2024-03-17 12:48:49
LastEditors: tany 1767774755@qq.com
LastEditTime: 2024-04-01 14:10:53
FilePath: /Chinese-CLIP/app_demo/test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import json
import sys
sys.path.append("/home/ubuntu/GITHUG/Chinese-CLIP")
from cn_clip.clip.model import CLIP,convert_weights
import torch
from cn_clip.clip import load_from_name, available_models






model, preprocess = load_from_name("ViT-B-16", device="cuda", download_root='./')
checkpoint = torch.load("/home/ubuntu/DataSet/BS/experiments/phonePicture/checkpoints/epoch85.pt", map_location='cuda')
sd = checkpoint["state_dict"]
if next(iter(sd.items()))[0].startswith('module'):
    sd = {k[len('module.'):]: v for k, v in sd.items() if "bert.pooler" not in k}
model.load_state_dict(sd)

import faiss
import numpy as np

import sqlite3

def load_features_from_db(database_file):
    with sqlite3.connect(database_file) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT imageFeatures FROM images")
        rows = cursor.fetchall()
    return rows

import io
import torch

def deserialize_features(rows):
    features = []
    for row in rows:
        tensor_bytes = row[0]  # 假设特征存储在行的第一列
        feature_tensor = torch.load(io.BytesIO(tensor_bytes))
        feature_np = feature_tensor.cpu().detach().numpy()  # 先移动到CPU，再转换为NumPy数组
        features.append(feature_np)
    return features

import numpy as np

def combine_features(features):
    return np.vstack(features)

def build_faiss_index(feature_vectors):
    dimension = feature_vectors.shape[1]  # 特征向量的维度
    index = faiss.IndexFlatL2(dimension)  # 使用L2距离创建索引
    feature_vectors = feature_vectors.astype('float32')
    index.add(feature_vectors)  # 将特征向量添加到索引中
    return index


def search_faiss_index(index, query_vector, k):
    distances, indices = index.search(query_vector, k)  # 执行搜索
    return distances, indices

def get_index_from_db(database_file):
    with sqlite3.connect(database_file) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT imageFeatures FROM images")
        rows = cursor.fetchall()
        features=[]
        for row in rows:
            tensor_bytes = row[0]  # 假设特征存储在行的第一列
            feature_tensor = torch.load(io.BytesIO(tensor_bytes))
            feature_np = feature_tensor.cpu().detach().numpy()  # 先移动到CPU，再转换为NumPy数组
            features.append(feature_np)
        feature_vectors=np.vstack(features)
        dimension = feature_vectors.shape[1]  # 特征向量的维度
        index = faiss.IndexFlatIP(dimension)  # 使用内积距离创建索引
        feature_vectors = feature_vectors.astype('float32')
        index.add(feature_vectors)  # 将特征向量添加到索引中
        return index

# 假设 db_features 是一个包含数据库中所有图像特征的NumPy数组
# 假设 search_feature 是你要搜索的图像特征，也是一个NumPy数组



database_file = 'database.db'
# Faisssearch=Index_search(database_file=database_file)

# # 从数据库加载特征向量
# rows = load_features_from_db(database_file)

# # 反序列化特征向量
# features = deserialize_features(rows)

# # 将特征向量列表转换为一个NumPy数组
# feature_vectors = combine_features(features)



# 现在，feature_vectors 就是一个形状为 (n_features, dim) 的NumPy数组，
# 你可以用它来构建FAISS索引
# index = build_faiss_index(feature_vectors)

index=get_index_from_db(database_file)

def get_images(index, query_vector,k):
    distances, indices = index.search(query_vector, k)
    # distances, indices = search_faiss_index(index, search_feature_np, 5)
    import base64
    with sqlite3.connect(database_file) as conn:
        images=[]
        cursor = conn.cursor()
        for inde in indices[0]:
            cursor.execute(f"SELECT imageBase64 FROM images WHERE id={inde}")
            row = cursor.fetchone()
            image_base64 = row[0]
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            image.show()
            images.append(image)
        return images
from PIL import Image
img = preprocess(img=Image.open("app_demo/demo/IMG20190205000215.jpg")).unsqueeze(0).to("cuda")
search_feature=model.encode_image(img).detach().cpu().numpy()
print(search_feature)
# 转换搜索特征为合适的形状（假设search_feature是一个一维数组）
search_feature_np = search_feature.astype("float32")
print(search_feature_np.shape)

get_images(index,search_feature_np,5)


# 使用FAISS索引来找到最相似的5个特征
# distances, indices = index.search(search_feature_np, 5)
# # distances, indices = search_faiss_index(index, search_feature_np, 5)

# import base64
# with sqlite3.connect(database_file) as conn:
#     images=[]
#     cursor = conn.cursor()
#     for inde in indices[0]:
#         cursor.execute(f"SELECT imageBase64 FROM images WHERE id={inde}")
#         row = cursor.fetchone()
#         image_base64 = row[0]
#         image_data = base64.b64decode(image_base64)
#         image = Image.open(io.BytesIO(image_data))
#         image.show()
#         images.append(image)



# # Faisssearch.get_images(search_feature,k=5)
# distances, indices = search_faiss_index(index, search_feature_np, 5)
# print(distances, indices)
