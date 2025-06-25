import torch
from PIL import Image
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor
from torch.nn.functional import cosine_similarity

# 加载模型和预处理器
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# 是否使用GPU
device = "mps:0"
model.to(device)

# 预处理图像的函数
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    return inputs['pixel_values'].to(device)

# 计算图像之间的余弦相似度
def compute_image_similarity(img1_path, img2_path):
    img1 = preprocess_image(img1_path)
    img2 = preprocess_image(img2_path)

    with torch.no_grad():
        img1_features = model.get_image_features(img1)
        img2_features = model.get_image_features(img2)

        # 归一化
        img1_features = img1_features / img1_features.norm(dim=-1, keepdim=True)
        img2_features = img2_features / img2_features.norm(dim=-1, keepdim=True)

        similarity = cosine_similarity(img1_features, img2_features)
        return similarity.item()

# 示例使用
if __name__ == "__main__":

    defect_paths = ["./357_arching.png", "./242_particle.png", "313_residue.png", "329_peeling.png"]
    gen_paths = ["357_gen.jpg", "242_gen.jpg", "313_gen.jpg", "329_gen.jpg"]
    gt_paths = ["357_gt.bmp", "242_gt.bmp", "313_gt.bmp", "329_gt.bmp"]

    scores = []
    for defect_path, gen_path in zip(defect_paths, gen_paths):
        similarity_score = compute_image_similarity(defect_path, gen_path)
        scores.append(similarity_score)
    print(f"Generated similarity score: {scores}")

    scores = []
    for defect_path, gen_path in zip(defect_paths, gt_paths):
        similarity_score = compute_image_similarity(defect_path, gen_path)
        scores.append(similarity_score)
    print(f"\n GT similarity score: {scores}")

    scores = []
    for defect_path, gen_path in zip(gen_paths, gt_paths):
        similarity_score = compute_image_similarity(defect_path, gen_path)
        scores.append(similarity_score)
    print(f"\n Gen/GT similarity score: {scores}")
