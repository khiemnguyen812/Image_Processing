from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import numpy as np
import sys

# Hàm tăng độ sáng
def brighten_image(image, scale):
    image = np.array(image)
    image=image.astype(int)
    
    #Tăng độ sáng hình ảnh
    image+=int(scale*255)
    image[image>255]=255
    
    return image.astype(np.uint8)
        
# Hàm tăng độ tương phản
def contrast_image(image,scale):
    image = np.array(image)
    image=image.astype(float)
    
    # Tăng độ tương phản
    image *= float(1 + scale) ** 2

    image[image > 255] = 255
    image[image < 0] = 0

    image = image.astype(np.uint8)
    return image

# Lật ảnh ngang-dọc với mode (int)
# Chế độ lật ảnh (0: lật theo chiều dọc, 1: lật theo chiều ngang).
def flip_image(image, mode):
    if(mode not in [0,1]): 
        print("Mode is wrong value.")
        sys.exit()
    # Chuyển đối tượng image thành một mảng numpy
    image_array = np.array(image)

    # Lấy kích thước ban đầu của ảnh (h, w)
    dim = image_array.shape

    # Lật ảnh theo chiều dọc, chiều ngang, hoặc cả hai chiều
    flipped_image_array = np.flip(image_array, mode)

    # Chuyển mảng numpy về đối tượng PIL Image
    flipped_image_pil = Image.fromarray(flipped_image_array)

    return flipped_image_pil

# Hàm chỉnh ảnh xám
def grey_image(image):
    alpha = 0.299
    beta = 0.587
    gamma = 0.114
    image = np.array(image)
    image = image.astype(float)
    
    gray_values = np.dot(image, np.array([alpha, beta, gamma]) / (alpha + beta + gamma))
    
    # Tạo mảng màu xám bằng cách lặp lại giá trị cho mỗi kênh màu
    gray_image = np.zeros_like(image)
    gray_image[:, :, 0] = gray_values
    gray_image[:, :, 1] = gray_values
    gray_image[:, :, 2] = gray_values

    gray_image = gray_image.astype(np.uint8)
    return gray_image

# Hàm chỉnh màu sepia
def sepia_image(image):
    # Định nghĩa ma trận chuyển đổi màu sepia
    sepia_matrix = np.array([[0.393, 0.769, 0.189],
                            [0.349, 0.686, 0.168],
                            [0.272, 0.534, 0.131]])

    # Chuyển đối tượng image thành mảng numpy và chuyển đổi thành kiểu dữ liệu float
    image = np.array(image)
    image = image.astype(float)

    # Thực hiện phép nhân giữa ma trận sepia_matrix và mảng image để chuyển đổi màu
    sepia_image = np.dot(image, sepia_matrix.T)

    # Giới hạn giá trị các kênh màu trong khoảng 0-255
    sepia_image = np.clip(sepia_image, 0, 255)

    # Chuyển mảng numpy về đối tượng PIL Image và định dạng kiểu dữ liệu thành uint8
    sepia_image = sepia_image.astype(np.uint8)
    sepia_image = Image.fromarray(sepia_image)

    return sepia_image

# Hàm tăng độ sắc nét 
def sharpen_image(image, amount=0.05):
    kernel = np.array([[0, -amount, 0], [-amount, 1 + 4 * amount, -amount], [0, -amount, 0]])
    
    image_array = np.array(image)
    
    if len(image_array.shape) == 3:  # Ảnh màu
        image_result = np.zeros_like(image_array)
        for channel in range(image_array.shape[2]):
            image_result[:, :, channel] = convolve2D(image_array[:, :, channel], kernel)
    elif len(image_array.shape) == 2:  # Ảnh đen trắng
        image_result = convolve2D(image_array, kernel)
    else:
        raise ValueError("Unsupported image shape.")
    
    # Giới hạn giá trị pixel trong khoảng 0 đến 255
    image_result = np.clip(image_result, 0, 255)
    
    image_result = image_result.astype(np.uint8)
    return image_result

# Hàm làm mờ ảnh 
def blur_image(image, amount=1.0):
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size)) * (amount / (kernel_size ** 2))
    
    image_array = np.array(image)
    
    if len(image_array.shape) == 3:  # Ảnh màu
        image_result = np.zeros_like(image_array)
        for channel in range(image_array.shape[2]):
            image_result[:, :, channel] = convolve2D(image_array[:, :, channel], kernel)
    elif len(image_array.shape) == 2:  # Ảnh đen trắng
        image_result = convolve2D(image_array, kernel)
    else:
        raise ValueError("Unsupported image shape.")
    
    # Giới hạn giá trị pixel trong khoảng 0 đến 255
    image_result = np.clip(image_result, 0, 255)
    
    image_result = image_result.astype(np.uint8)
    return image_result

# Hàm thực hiện phép tích chập giữa một hình ảnh và một kernel (bộ lọc).
def convolve2D(image, kernel):
    kernelDim = kernel.shape[0]
    padSize = kernelDim // 2
    
    image_padded = np.pad(image, ((padSize, padSize), (padSize, padSize)), mode='edge')
    image_result = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image_result[i, j] = np.sum(image_padded[i:i+kernelDim, j:j+kernelDim] * kernel)
            
    return image_result

# Hàm cắt ảnh ở trung tâm
def crop_and_resize_center_image(image, scale=0.5):
    image_array = np.array(image)
    height, width = image_array.shape[:2]

    new_width = int(width * scale)
    new_height = int(height * scale)

    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height

    cropped_image_array = image_array[top:bottom, left:right]

    cropped_image = Image.fromarray(cropped_image_array)

    return cropped_image

# Hàm cắt theo khung hình tròn
def circle_frame(image, mode=0):
    image_array = np.array(image)
    height, width = image_array.shape[:2]

    center = np.array([height/2, width/2])
    radius = mode * max(height/2, width/2) + (1 - mode) * min(height/2, width/2)

    x, y = np.ogrid[:height, :width]

    mask_circle = (x - center[0]) ** 2 + (y - center[1]) ** 2 > radius ** 2
    image_array[mask_circle] = np.zeros(image_array.shape[2])

    return Image.fromarray(image_array)

# Hàm cắt ảnh theo 2 hình elip chồng nhau
def ellipse_frame(image, scale=1.25, angle=45):
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
    rotated_mask_ellipse = np.ones((new_edge, new_edge), dtype=bool)
    rotated_mask_ellipse[x_new, y_new] = mask_ellipse[x_ellipse, y_ellipse]

    # Tính toán vị trí để đưa hình elip chéo vào giữa ảnh
    c = (rotated_mask_ellipse.shape[0] - height) // 2
    r = (rotated_mask_ellipse.shape[1] - width) // 2
    rotated_mask_ellipse = rotated_mask_ellipse[c:c+height, r:r+width]

    temp = np.ones((height, width), dtype=bool)
    temp[:rotated_mask_ellipse.shape[0], :rotated_mask_ellipse.shape[1]] = rotated_mask_ellipse

    # Áp dụng mặt nạ với ảnh
    image_array[temp] = np.zeros(image_array.shape[2])

    return Image.fromarray(image_array)

# Hàm giúp save ảnh
def save_image_as_png(img, file_name):
    try:
        img = Image.fromarray(img)  # Convert the NumPy array to Image object
        img.save(f"{file_name}.png", format="PNG")
        print(f"Image saved as '{file_name}.png'")
    except Exception as e:
        print(f"Error: An unexpected error occurred while saving the image: {e}")


# Hàm để mở ảnh from
SUPPORTED_IMAGE_FORMATS = ["PNG", "JPEG", "GIF", "BMP", "ICO", "TIFF", "WEBP", "JPG"]
def open_image_by_name(image_name):
    try:
        for format in SUPPORTED_IMAGE_FORMATS:
            image_path = f"{image_name}.{format.lower()}"
            try:
                img = Image.open(image_path,'r')
                return img
            except Exception:
                pass
        
        print(f"Error: Unable to open the image '{image_name}' in any supported format.")
        return None
    except Exception as e:
        print(f"Error: An unexpected error occurred while opening the image '{image_name}': {e}")
        return None
    
def main():
    # Nhập tên tập tin ảnh
    image_name = input("Nhập tên tập tin ảnh (không cần đuôi mở rộng): ")

    # Mở ảnh từ tên tập tin đã nhập
    image = open_image_by_name(image_name)

    # Kiểm tra ảnh đã mở thành công chưa
    if image is None:
        return

    print("Danh sách chức năng xử lý ảnh:")
    print("1. Tăng độ sáng")
    print("2. Tăng độ tương phản")
    print("3. Chuyển sang ảnh xám")
    print("4. Lật ảnh (0: lật theo chiều dọc, 1: lật theo chiều ngang)")
    print("5. Chỉnh màu sepia")
    print("6. Làm sắc nét ảnh")
    print("7. Làm mờ ảnh")
    print("8. Cắt ảnh theo khung tròn")
    print("9. Cắt ảnh theo hai hình elip chéo nhau")
    print("10. Cắt ảnh ở trung tâm.")
    print("0. Thực hiện tất cả chức năng trên")

   
    choice = int(input("Nhập số tương ứng với chức năng xử lý ảnh: "))
    if choice not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        print("Chọn không hợp lệ. Kết thúc chương trình.")
        return

    if choice == 0:
        # Thực hiện tất cả chức năng và lưu từng phiên bản ảnh với tên file đầu ra tương ứng
        brightened_image = brighten_image(image, 0.5)
        save_image_as_png(brightened_image, f"{image_name}_brightened")

        contrasted_image = contrast_image(image, 0.5)
        save_image_as_png(contrasted_image, f"{image_name}_contrasted")

        gray_image = grey_image(image)
        save_image_as_png(np.array(gray_image), f"{image_name}_grey")

        flipped_horizontal_image = flip_image(image, 1)
        save_image_as_png(np.array(flipped_horizontal_image), f"{image_name}_flipped_horizontal")

        flipped_vertical_image = flip_image(image, 0)
        save_image_as_png(np.array(flipped_vertical_image), f"{image_name}_flipped_vertical")

        sepia_toned_image = sepia_image(image)
        save_image_as_png(np.array(sepia_toned_image), f"{image_name}_sepia")

        sharpened_image = sharpen_image(image)
        save_image_as_png(sharpened_image, f"{image_name}_sharpened")

        blurred_image = blur_image(image)
        save_image_as_png(blurred_image, f"{image_name}_blurred")
        
        circle_image=circle_frame(image)
        save_image_as_png(np.array(circle_image), f"{image_name}_circle_frame")

        ellipse_image=ellipse_frame(image)
        save_image_as_png(np.array(ellipse_image), f"{image_name}_ellipse_frame")
        
        crop_image=crop_and_resize_center_image(image)
        save_image_as_png(np.array(crop_image), f"{image_name}_crop")
        
        print("Đã thực hiện tất cả chức năng và lưu các phiên bản ảnh.")
        
        # Ảnh brightened
        fig, axes = plt.subplots(3, 4, figsize=(30, 20))
        axes[0, 0].set_title(f'Original image')
        axes[0, 0].imshow(np.array(image))

        # Hiển thị các ảnh sau khi xử lý
        axes[0, 1].set_title(f'Brightened image')
        axes[0, 1].imshow(brightened_image)

        axes[0, 2].set_title(f'Contrasted image')
        axes[0, 2].imshow(contrasted_image)

        axes[0, 3].set_title(f'Gray image')
        axes[0, 3].imshow(gray_image, cmap='gray')

        axes[1, 0].set_title(f'Flipped horizontal image')
        axes[1, 0].imshow(flipped_horizontal_image)

        axes[1, 1].set_title(f'Flipped vertical image')
        axes[1, 1].imshow(flipped_vertical_image)

        axes[1, 2].set_title(f'Sepia-toned image')
        axes[1, 2].imshow(sepia_toned_image)

        axes[1, 3].set_title(f'Sharpened image')
        axes[1, 3].imshow(sharpened_image)

        axes[2, 0].set_title(f'Blurred image')
        axes[2, 0].imshow(blurred_image)
        
        axes[2, 1].set_title(f'Circle frame image')
        axes[2, 1].imshow(circle_image)

        axes[2, 2].set_title(f'Ellipse frame image')
        axes[2, 2].imshow(ellipse_image)
    
        axes[2,3].set_title(f'Crop image')
        axes[2, 3].imshow(crop_image)
    else:
        name_output=''
        # Thực hiện chức năng tương ứng và lưu ảnh với tên file đầu ra tương ứng
        if choice == 1:
            scale = float(input("Nhập giá trị tăng độ sáng (0-1): "))
            if(scale<0 or scale>1): 
                print("Giá trị không hợp lệ.")
                return
            processed_image = brighten_image(image, scale)
            name_output=f"{image_name}_brightened"
        elif choice == 2:
            scale = float(input("Nhập giá trị tăng độ tương phản (0-1): "))
            if(scale<0 or scale>1): 
                print("Giá trị không hợp lệ.")
                return
            processed_image = contrast_image(image, scale)
            name_output=f"{image_name}_contrasted"
        elif choice == 3:
            processed_image = grey_image(image)
            name_output= f"{image_name}_grey"
        elif choice == 4:
            mode = int(input("Nhập chế độ lật ảnh (1: lật theo chiều ngang, -1: lật theo chiều dọc): "))
            processed_image = flip_image(image, mode)
            if(mode==1): 
                name_output=f"{image_name}_flipped_horizontal"
            if(mode==-1): name_output=f"{image_name}_flipped_vertical"
            if(mode!=1 and mode!=1): 
                print("Giá trị không hợp lệ.")
                return
        elif choice == 5:
            processed_image = sepia_image(image)
            name_output=f"{image_name}_sepia"
        elif choice == 6:
            processed_image = sharpen_image(image)
            name_output=f"{image_name}_sharpened"
        elif choice == 7:
            processed_image = blur_image(image)
            name_output=f"{image_name}_blurred"
        elif choice == 8:
            processed_image = circle_frame(image)
            name_output=f"{image_name}_circle_frame"
        elif choice == 9:
            processed_image=ellipse_frame(image)
            name_output=f"{image_name}_ellipse_frame"
        elif choice==10:
            processed_image=crop_and_resize_center_image(image)
            name_output=f"{image_name}_crop"
            
        processed_image = np.array(processed_image)

        save_image_as_png(processed_image, name_output)
        
        demo,axis=plt.subplots(1,2,figsize=(14,10))
        axis[0].set_title(f'Original image: ')
        axis[0].imshow(image)
        axis[1].set_title(f'New image: ')
        axis[1].imshow(processed_image)
    
        
        print(f"Đã thực hiện chức năng {choice} và lưu ảnh với tên file đầu ra.")

if __name__ == "__main__":
    main()
    plt.show()
