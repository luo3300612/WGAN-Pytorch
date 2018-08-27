from PIL import Image
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', required=True, help='show me where your data is, you mother fucker')
    parser.add_argument('--optfile', required=True, help='show me where to output.')

    opt = parser.parse_args()

    data_path = opt.dataroot
    opt_file = opt.optfile
    data_path = data_path.replace('\\', '/')
    opt_file = opt_file.replace('\\', '/')

    try:
        os.mkdir(opt.optfile)
    except OSError:
        pass

    img_names = os.listdir(data_path)

    if data_path[-1] != '/':
        data_path += '/'

    if opt_file[-1] == '/':
        opt_file = opt_file[:-1]

    print(opt_file)

    print(data_path)

    total = len(img_names)
    index = 0
    for img_name in img_names:
        index += 1
        im = Image.open(data_path + img_name)
        out = im.transpose(Image.FLIP_LEFT_RIGHT)
        im.save("%s/%sA.png" % (opt_file, img_name.split(".")[0]))
        out.save("%s/%sB.png" % (opt_file, img_name.split(".")[0]))
        print("%d:%.2f%%" % (index, index / total * 100))
    print("done")
