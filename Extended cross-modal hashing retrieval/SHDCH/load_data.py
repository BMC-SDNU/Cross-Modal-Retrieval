import scipy.io as sio


def loading_data(path):

    image_path = path + "image.mat"
    label_path = path + "label.mat"
    tag_path = path + "tag.mat"

    images = sio.loadmat(image_path)['Image']   # 19862,224,224,3
    tags = sio.loadmat(tag_path)['Tag']     # 19862,2685
    labels = sio.loadmat(label_path)["Label"]    # 19862,35

    return images, tags, labels


if __name__ == '__main__':
    path = './DataSet/Ssense/'
    images, tags, labels = loading_data(path)
    print(images.shape)
    print(tags.shape)
    print(labels.shape)
