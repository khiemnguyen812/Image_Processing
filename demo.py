import numpy as np
from PIL import Image

def interlaceEllipseFrame(image, scale=1.25, angle=45):
    image_array = np.array(image)
    height, width = image_array.shape[:2]

    # Tính toán các thông số cho việc xoay
    angle_rad = np.radians(angle)
    sine = np.sin(angle_rad)
    cosine = np.cos(angle_rad)

    # Xác định kích thước cạnh của ảnh hình elip
    edge = int(min(width, height) * scale)
    # Xác định kích thước mới sau khi xoay ảnh hình elip
    new_edge = int(np.ceil(abs(edge * cosine) + abs(edge * sine)))

    # Tạo 2 mặt nạ elip
    x_ellipse, y_ellipse = np.ogrid[:edge, :edge]
    smallRadius = edge / 4
    largeRadius = edge / 2

    mask_ellipse1 = (x_ellipse - edge / 2) ** 2 / largeRadius ** 2 + (y_ellipse - edge / 2) ** 2 / smallRadius ** 2 > 1
    mask_ellipse2 = (x_ellipse - edge / 2) ** 2 / smallRadius ** 2 + (y_ellipse - edge / 2) ** 2 / largeRadius ** 2 > 1
    mask_ellipse = mask_ellipse1 & mask_ellipse2

    # Tính toán các vị trí mới tương ứng với x_ellipse, y_ellipse
    y = edge / 2 - y_ellipse
    x = edge / 2 - x_ellipse
    x_new = (new_edge / 2 - np.ceil(x * cosine + y * sine)).astype(int)
    y_new = (new_edge / 2 - np.ceil(-x * sine + y * cosine)).astype(int)

    # Chuyển đổi thành mặt nạ mới sau khi xoay
    rotated_mask_ellipse = np.zeros((new_edge, new_edge), dtype=bool)
    rotated_mask_ellipse[x_new, y_new] = mask_ellipse[x_ellipse, y_ellipse]

    # Tính toán vị trí cắt trong ảnh gốc
    cx, cy = width // 2, height // 2
    x_start, x_end = cx - edge // 2, cx + edge // 2
    y_start, y_end = cy - edge // 2, cy + edge // 2

    # Tạo mặt nạ trên ảnh gốc
    result_image = np.copy(image_array)
    result_image[y_start:y_end, x_start:x_end] = np.where(
        np.expand_dims(rotated_mask_ellipse, axis=-1),  # Expand dimensions of the mask to match the number of color channels
        image_array[y_start:y_end, x_start:x_end],  # Keep the original pixels where the mask is True
        0,  # Set the pixels to black where the mask is False
    )

    return Image.fromarray(result_image)

# Sử dụng hàm với ảnh đầu vào (giả sử có thư viện PIL được import và ảnh là "input_image.jpg")
input_image = Image.open("CR7.png")
output_image = interlaceEllipseFrame(input_image)
output_image.show()
