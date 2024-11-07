from PIL import Image

# 打开原图和掩码
image = Image.open('dataset/VGA/test/images/rgb_test_0.png').convert("RGB")
mask = Image.open('dataset/VGA/test/mask/mask_test_0.png').convert("L")

# 创建一个空白的 RGBA 图像，作为最终的抠图结果
output = Image.new("RGB", image.size, (255, 255, 255))

# 将原图和掩码合并
output = Image.composite(image, output, mask)

# 保存或展示抠图结果
output.save('rgb_mask.png')
#output.show()