import gradio as gr
import sys
import torch
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

def calculate_similarity(search_feature, db_feature):
    # 计算余弦相似度
    similarity = torch.dot(search_feature, db_feature) / (torch.norm(search_feature) * torch.norm(db_feature))
    return similarity.item()  # 返回相似度值


# Function to call the API for retrieving similar images
def retrieve_images(image, num):
    search_feature = model.processimg(Image.fromarray(image)).squeeze()
    
    with sqlite3.connect(database_file) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT imageBase64, imageFeatures FROM images")
        rows = cursor.fetchall()
        
        similarities = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for row in rows:
                base64_img = row[0]
                tensor_bytes = row[1]
                db_feature = torch.load(io.BytesIO(tensor_bytes)).squeeze()
                futures.append(executor.submit(calculate_similarity, search_feature, db_feature))
                
            for future, row in zip(concurrent.futures.as_completed(futures), rows):
                similarity = future.result()
                similarities.append((similarity, row[0]))
                
    similarities.sort(reverse=True)
    result = similarities[:num]
    images = [base64_to_image(re[1]) for re in result]
    return images

# Function to call the API for retrieving similar text
def retrieve_text(text,num):
    search_feature = model.processtxt(text).squeeze()
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()
    cursor.execute("SELECT imageBase64, imageFeatures FROM images")
    rows = cursor.fetchall()
    
    similarities = []
    for row in rows:
        base64_img = row[0]
        tensor_bytes = row[1]
        
        # 将二进制数据加载为张量
        db_feature = torch.load(io.BytesIO(tensor_bytes)).squeeze()
        
        # 计算检索文本特征与数据库中每个文本特征之间的相似度
        similarity = calculate_similarity(search_feature, db_feature)
        
        # 将相似度和文本添加到列表中
        similarities.append((similarity, base64_img))
    
    similarities.sort(reverse=True)
    result = similarities[:num]
    images=[]
    for re in result:
        img=base64_to_image(re[1])
        img=base64_to_image(re[1])
        images.append(img)
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
    fn=retrieve_images,
    inputs=[gr.components.Image(),gr.Number(5)],
    outputs=gr.components.Gallery(height=None,columns=4,preview=True),
    title="图搜图",
    description="图片搜索图片",
    submit_btn="检索",
    clear_btn="清除"
)

# Gradio Interface for retrieving similar text
retrieve_text_interface = gr.Interface(
    fn=retrieve_text,
    inputs=[gr.components.Textbox(lines=1, placeholder="Enter text here..."),gr.Number(5)],
    outputs=gr.components.Gallery(),
    title="文搜图",
    description="文本搜索图片",
    submit_btn="检索",
    clear_btn="清除"

)

# Launch the Gradio interfaces

with gr.TabbedInterface(
        [upload_images_interface, retrieve_images_interface, retrieve_text_interface],
        ["上传图片", "图搜图", "文搜图"]
    ) as demo:
        demo.launch()  # 启动标签页式界面