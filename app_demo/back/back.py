from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import Compose,Normalize,Resize,ToTensor,InterpolationMode,
from PIL import Image
from io import BytesIO
import os,lmdb,base64



def _convert_to_rgb(image):
    return image.convert('RGB')


class EvalImgDataset(Dataset):
    def __init__(self, lmdb_imgs, resolution=224):
        assert os.path.isdir(lmdb_imgs), "The image LMDB directory {} not exists!".format(lmdb_imgs)

        self.env_imgs = lmdb.open(lmdb_imgs, readonly=True, create=False, lock=False, readahead=False, meminit=False)
        self.txn_imgs = self.env_imgs.begin(buffers=True)
        self.cursor_imgs = self.txn_imgs.cursor()

        self.iter_imgs = iter(self.cursor_imgs)
        self.number_images = int(self.txn_imgs.get(key=b'num_images').tobytes().decode('utf-8'))

        self.transform = self._build_transform(resolution)

    def _build_transform(self, resolution):
        normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        return Compose([
                Resize((resolution, resolution), interpolation=InterpolationMode.BICUBIC),
                _convert_to_rgb,
                ToTensor(),
                normalize,
            ])

    def __len__(self):
        return self.number_images

    def __getitem__(self, idx):
        img_id, image_b64 = next(self.iter_imgs)
        if img_id == b"num_images":
            img_id, image_b64 = next(self.iter_imgs)

        img_id = img_id.tobytes()
        image_b64 = image_b64.tobytes()

        img_id = int(img_id.decode(encoding="utf8", errors="ignore"))
        image_b64 = image_b64.decode(encoding="utf8", errors="ignore")
        image = Image.open(BytesIO(base64.urlsafe_b64decode(image_b64))) # already resized
        image = self.transform(image)

        return img_id, image