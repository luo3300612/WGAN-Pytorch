import os
from PIL import Image


def dhash(img):
    resize_width = 9
    resize_height = 8

    downsizedImg = img.resize((resize_width, resize_height))
    grayscale_image = downsizedImg.convert("L")

    pixels = list(grayscale_image.getdata())

    difference = []
    for row in range(resize_height):
        row_start_index = row * resize_width
        for col in range(resize_width-1):
            left_pixel_index = row_start_index + col
            difference.append(pixels[left_pixel_index] > pixels[left_pixel_index + 1])

    decimal_value = 0
    hash_string = ""
    for index, value in enumerate(difference):
        if value:
            decimal_value += value * (2 ** (index%8))
        if index % 8 == 7:
            hash_string += str(hex(decimal_value)[2:].rjust(2, "0"))
            decimal_value = 0

    return hash_string

def img_in_list(img, hashlist):
    hash_of_img = dhash(img)
    if hash_of_img in hashlist:
        return True,""
    else:
        return False,hash_of_img

if __name__ == "__main__":
    dirs = os.listdir("./imgs/")
    hashlist = []

    try:
        os.makedirs("./result")
    except FileExistsError:
        pass
    duplicate = 0
    total = 0
    for dir_ in dirs:
        path = "./imgs/" + dir_
        imgs = os.listdir(path + "/")
        for img in imgs:
            try:
                im = Image.open(path + "/" + img)
            except OSError:
                continue
            ret = img_in_list(im, hashlist)
            total += 1
            if ret[0]:
                duplicate += 1
            else:
                hashlist.append(ret[1])
                im_ = im.convert("RGB")
                im_.save("./result/" + str(total-duplicate) + ".jpg")


            print(f"{dir_}:{total-duplicate}/{total}")

