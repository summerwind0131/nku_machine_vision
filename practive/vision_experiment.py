from pathlib import Path

import cv2
import numpy as np

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def save_image(filename:str, image: np.ndarray)->None:
    path=OUTPUT_DIR/filename
    success=cv2.imwrite(str(path),image)
    if not success:
        raise RuntimeError(f"无法保存图像: {path}")
    
def normalize_to_uint8(image:np.ndarray)->np.ndarray:
    normalized=cv2.normalize(
        image,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX
    )
    
    return normalized.astype(np.unit8)

def brightness_transform(gray:np.ndarray)->None:
    """灰度级变换"""

    # 反色变换
    negative=255-gray

    # Gamma变换
    gamma=0.5
    normalized=gray.astype(np.float32)/255.0
    gamme_image=255.0*np.power(normalized,gamma)
    gamma_image=np.clip(gamma_image,0,255).astype(np.uint8)

    #线性亮度增强
    linear=1.3*gray.astype(np.float32)+20
    linear=np.clip(linear,0,255).astype(np.uint8)

    #阈值变换
    threshold_value=128
    _,binary=cv2.threshold(
        gray,
        threshold_value,
        255,
        cv2.THRESH_BINARY
    )

    #对数变换
    # 5. 对数变换：s = c log(1 + r)
    log_image = np.log1p(gray.astype(np.float32))
    log_image = normalize_to_uint8(log_image)

    save_image("01_negative.jpg", negative)
    save_image("02_gamma.jpg", gamma_image)
    save_image("03_linear.jpg", linear)
    save_image("04_binary.jpg", binary)
    save_image("05_log.jpg", log_image)


def prewitt_experiment(gray:np.ndarray)->None:
    prewitt=np.array([
        [1,1,1],
        [0,0,0],
        [-1,-1,-1],
    ],
    dtype=np.float32
    )

    correlation=cv2.filter2D(
        gray,
        ddepth=cv2.CV_32F,
        kernel=prewitt,
        borderType=cv2.BORDER_CONSTANT
    )

    # 严格数学卷积：先将核旋转 180°
    flipped_kernel = np.flip(prewitt, axis=(0, 1)).copy()

    convolution = cv2.filter2D(
        gray,
        ddepth=cv2.CV_32F,
        kernel=flipped_kernel,
        borderType=cv2.BORDER_CONSTANT,
    )

    correlation_display = normalize_to_uint8(correlation)
    convolution_display = normalize_to_uint8(convolution)

    # 边缘强度通常观察绝对值
    correlation_abs = cv2.convertScaleAbs(correlation)
    convolution_abs = cv2.convertScaleAbs(convolution)

    save_image("06_prewitt_correlation.jpg", correlation_display)
    save_image("07_prewitt_convolution.jpg", convolution_display)
    save_image("08_prewitt_correlation_abs.jpg", correlation_abs)
    save_image("09_prewitt_convolution_abs.jpg", convolution_abs)

    difference = np.max(np.abs(convolution + correlation))

    print("Prewitt 原核：")
    print(prewitt)
    print("翻转后的卷积核：")
    print(flipped_kernel)
    print(f"max|convolution + correlation| = {difference:.6f}")
    print("这个 Prewitt 核是反对称的，因此卷积和互相关通常只差一个负号。")



