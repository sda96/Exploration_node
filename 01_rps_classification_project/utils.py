import glob
import numpy as np
from PIL import Image
import imgaug as ia
import imgaug.augmenters as iaa


def data_path_load(data_path:"str"):
    """
    데이터 불러오는 디렉토리 구조
    - rock_paper_scissors
    -- rock
        --- 1.jpg
        --- 2.jpg
            ...
            ..
        --- 100.jpg
    -- paper
        --- 1.jpg
        --- 2.jpg
            ...
            ..
    """
    
    # 입력된 경로에 클래스 명을 가져옵니다.
    directories = glob.glob(data_path)
    class_name = list(map(lambda x: x.split("/")[-1], directories))
    
    # 각 클래스 명과 jpg 경로를 mapping 합니다.
    name_jpg = dict()
    for name in class_name:
        class_path = data_path.replace("*", name) + "/*"
        name_jpg.update({name : glob.glob(class_path)})    
        
    return name_jpg

def load_data(data_paths):
    x_data = []
    y_data = []

    for path in data_paths:
        name_jpg = data_path_load(path)


        tmp_x_data = []
        tmp_y_data = []
        width, height = 224, 224
        target_size = (width, height)
        for idx, name in enumerate(name_jpg):
            # jpg 파일을 하나씩 불러와서 28x28로 이미지 크기 재조정합니다.
            tmp = [np.array(Image.open(jpg).resize(target_size, Image.ANTIALIAS)) for jpg in name_jpg[name]]
            # 클래스 종류별로 리스트에 넣습니다.
            tmp_x_data.append(tmp)
            tmp_y_data.append([idx]*len(tmp_x_data[idx]))

        # 종류별로 나뉘어진 차원을 통일시켜줍니다.
        tmp_x_data = np.array(tmp_x_data)
        tmp_y_data = np.array(tmp_y_data)
        tmp_x_data = tmp_x_data.reshape(-1, width, height, 3)
        tmp_y_data = tmp_y_data.reshape(-1, 1)
        x_data.append(tmp_x_data)
        y_data.append(tmp_y_data)

    total_x_data = np.array(x_data).reshape(-1, width, height, 3)
    total_y_data = np.array(y_data).reshape(-1,1)
    return total_x_data, total_y_data


def imgaug(x_data):    
    seq = iaa.Sequential([
        iaa.SomeOf(
            (1,3),
            [iaa.Multiply((0.5, 2.0)),
             iaa.Fliplr(0.5), # 수평 뒤집기
             iaa.Crop(percent=(0, 0.1)), # random crops
             iaa.Affine(
                 rotate=(-10, 10),
             )]
        )
    ], random_order=True) # apply augmenters in random order
    
    return seq(images = x_data)
