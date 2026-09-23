from PIL import Image


def add_alpha_channel(img):
    img = Image.open(img)
    img = img.convert('RGBA')
    # 更改图像透明度
    # factor = 0.7
    # img_blender = Image.new('RGBA', img.size, (0, 0, 0, 0))
    # img = Image.blend(img_blender, img, factor)
    return img


def image_together(image, layer1, layer2, save_path, save_name):
    layer1 = layer1
    layer2 = layer2
    base = image
    bands = list(layer1.split())
    heigh, width = layer1.size
    for i in range(heigh):
        for j in range(width):
            r, g, b, a = layer1.getpixel((i, j))
            if r == 0:
                layer1.putpixel((i, j), (0, 0, 0, 0))  # 背景透明显示
            else:
                layer1.putpixel((i, j), (0, 0, 256, 200))  # 非背景区域显示为红色
    layer2.paste(layer1, (0, 0), layer1)  # 贴图操作
    base = image
    bands = list(layer2.split())
    heigh, width = layer2.size
    for i in range(heigh):
        for j in range(width):
            r, g, b, a = layer2.getpixel((i, j))
            if r == 0:
                layer2.putpixel((i, j), (0, 0, 0, 0))
            elif r == 128 and g == 128 and b == 128:
                layer2.putpixel((i, j), (128, 128, 128, 200))
            else:
                layer2.putpixel((i, j), (255, 0, 0, 200))
    base.paste(layer2, (0, 0), layer2)  # 贴图操作
    base.save(save_path + "/" + save_name + ".png")  # 图片保存


if __name__ == "__main__":
    image = add_alpha_channel(r"")
    layer1 = add_alpha_channel(r"")
    layer2 = add_alpha_channel(r"")
    image_together(image, layer1, layer2, r"", "")

