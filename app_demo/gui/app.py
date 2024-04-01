import gradio as gr
import sys
import torch
import faiss
import numpy as np
sys.path.append("/home/ubuntu/GITHUG/Chinese-CLIP")
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
# print("Available models:", available_models())  
# Available models: ['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50']
from PIL import Image
import base64
import io
import sqlite3
import concurrent.futures
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read())
    return encoded_string.decode('utf-8')

def base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image

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


class clip_model:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = load_from_name("ViT-B-16", device=self.device, download_root='./')
        checkpoint = torch.load("/home/ubuntu/DataSet/BS/experiments/phonePicture/checkpoints/epoch85.pt", map_location='cuda')
        sd = checkpoint["state_dict"]
        if next(iter(sd.items()))[0].startswith('module'):
            sd = {k[len('module.'):]: v for k, v in sd.items() if "bert.pooler" not in k}
        self.model.load_state_dict(sd)
        self.model.eval()

    def processimg(self,image):
        with torch.no_grad():
            image = self.preprocess(image).unsqueeze(0).to(self.device)
            image_features = self.model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True) 
            return image_features
    
    
    def processtxt(self,txt):
        with torch.no_grad():
            text = clip.tokenize([txt]).to(self.device)
            text_features = self.model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)  
            return text_features
    
model = clip_model()
import os

database_file = 'database.db'

if not os.path.exists(database_file):
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()

    # 创建数据表
    cursor.execute('''CREATE TABLE images
                      (id INTEGER PRIMARY KEY AUTOINCREMENT,
                       imageBase64 TEXT,
                       imageFeatures BLOB)''')
    conn.commit()
    conn.close()
    print("数据库和数据表创建成功。")
else:
    print("数据库已经存在，跳过创建。")

faiss_index = get_index_from_db(database_file)
print(f"从{database_file}建立faiss索引，共{faiss_index.ntotal}个特征")

def get_search_result(faiss_index, query_vector,k):
    distances, indices = faiss_index.search(query_vector, k)
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
            # image.show()
            images.append(image)
        return images
    


# Function to process uploaded images
def process_images(images):  
    try:
        conn = sqlite3.connect(database_file)
        cursor = conn.cursor()
        if images is None or len(images) == 0:
            return "请选择图片"
        
        for img_path in images:
            try:
                with Image.open(img_path[0]) as img:
                    base64_img = image_to_base64(img_path[0])
                    image_features = model.processimg(img)
                    tensor_bytes = io.BytesIO()
                    torch.save(image_features, tensor_bytes)
                    cursor.execute("INSERT INTO images (imageBase64, imageFeatures) VALUES (?, ?)", 
                                   (base64_img, sqlite3.Binary(tensor_bytes.getvalue())))
            except (IOError, OSError) as e:
                return f"Error processing image: {e}"
        conn.commit()
        return "success"
    except Exception as e:
        return f"Error: {e}"
    finally:
        if conn:
            conn.close()


def retrieve_images_optimized(image, num):
    search_feature = model.processimg(Image.fromarray(image)).squeeze().detach().cpu().numpy()
    # print(search_feature)
    # 转换搜索特征为合适的形状（假设search_feature是一个一维数组）
    search_feature_np = search_feature.astype("float32").reshape(1, 512)
    # print(search_feature_np.shape)
    images=get_search_result(faiss_index,search_feature_np,num)
    return images

def retrieve_text_optimized(text,num):
    search_feature = model.processtxt(text).squeeze().detach().cpu().numpy()
    search_feature_np = search_feature.astype("float32").reshape(1, 512)
    images=get_search_result(faiss_index,search_feature_np,num)
    return images




# Gradio Interface for uploading multiple images
upload_images_interface = gr.Interface(
    fn=process_images,
    inputs=gr.components.Gallery(),
    outputs="text",
    title="上传图片",
    description="上传图片到数据库",
    submit_btn="上传",
    clear_btn="清除"
)


# Gradio Interface for retrieving similar images
retrieve_images_interface = gr.Interface(
    fn=retrieve_images_optimized,
    inputs=[gr.components.Image(),gr.Number(5)],
    outputs=gr.components.Gallery(height=None,columns=5,preview=True),
    title="图搜图",
    description="图片搜索图片",
    submit_btn="检索",
    clear_btn="清除"
)

# Gradio Interface for retrieving similar text
retrieve_text_interface = gr.Interface(
    fn=retrieve_text_optimized,
    inputs=[gr.components.Textbox(lines=1, placeholder="Enter text here..."),gr.Number(5)],
    outputs=gr.components.Gallery(height=None,columns=5,preview=True),
    title="文搜图",
    description="文本搜索图片",
    submit_btn="检索",
    clear_btn="清除"

)

# Launch the Gradio interfaces

with gr.TabbedInterface(
        [ retrieve_text_interface,retrieve_images_interface,upload_images_interface, ],
        [ "文搜图","图搜图", "上传图片"]
    ) as demo:
        demo.launch()  # 启动标签页式界面