import sqlite3
import sqlite3
import json
# 连接到数据库
conn = sqlite3.connect('/home/ubuntu/GITHUG/BLIP2-Chinese/imagecaption.db')
cursor = conn.cursor()

# 从数据库中读取所有数据
cursor.execute("SELECT rowid, imagebase64, caption3 FROM images")
rows = cursor.fetchall()

# 划分数据集
total_samples = len(rows)
train_samples = int(total_samples * 0.8)
val_samples = int(total_samples * 0.1)
test_samples = total_samples - train_samples - val_samples

# 分割数据集
train_data = rows[:train_samples]
val_data = rows[train_samples:train_samples + val_samples]
test_data = rows[train_samples + val_samples:]

# 将数据写入TSV文件
def write_to_tsv(data, split):
    with open(f"{split}_imgs.tsv", 'w') as f:
        for row in data:
            rowid, imagebase64, caption3 = row
            # 将 imagebase64 写入文件，每行包含图片路径和图片base64，以 tab 隔开
            f.write(f"{rowid}\t{imagebase64}\n")

# 将数据写入 JSONL 文件
def write_to_jsonl(data, split):
    filename = f"{split}_texts.jsonl"
    with open(filename, 'w', encoding='utf-8') as file:
        for row in data:
            rowid, _, caption3 = row
            # 构建 JSON 对象
            json_obj = {
                "text_id": rowid,
                "text": caption3,
                "image_ids": [rowid]  # 这里假设图片ID与文本ID相同
            }
            # 将 JSON 对象写入文件
            file.write(json.dumps(json_obj, ensure_ascii=False) + '\n')




# 将训练数据写入 TSV 文件
write_to_tsv(train_data, "train")

# 将验证数据写入 TSV 文件
write_to_tsv(val_data, "valid")

# 将测试数据写入 TSV 文件
write_to_tsv(test_data, "test")


# 将训练数据写入 JSONL 文件
write_to_jsonl(train_data, "train")

# 将验证数据写入 JSONL 文件
write_to_jsonl(val_data, "valid")

# 将测试数据写入 JSONL 文件
write_to_jsonl(test_data, "test")

