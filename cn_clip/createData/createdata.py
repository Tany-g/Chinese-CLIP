import typing_extensions
import os
from PIL import Image
class pictureCreater():
    def __init__(self,iconPath: str) -> None:
        self.icon = self.__loadPicture(iconPath)      
        self.imageLis = []
        
    def __loadPicture(self,path: str)->None:
        assert os.path.isfile(path),f"file path {path} not exits"
        return Image.open(path)
    
    def LoadPic(self,path :str):
        self.imageLis.append(self.__loadPicture(path))


    def pixelCoordinates2RelativeCoordinates(self,pix_x,pix_y):
        pass


if __name__=="__main__":
    Creater = pictureCreater("cn_clip/createData/iconPic/icon.png")
    