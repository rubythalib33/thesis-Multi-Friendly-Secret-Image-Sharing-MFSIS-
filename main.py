from math import log10, sqrt
import cv2
import numpy as np


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def bit8to4(img):
    return (img//16).astype(np.uint8)


def bit4to8(img):
    return img*16


def hash_image(images):
    temp=np.zeros(images[0].shape)
    temp=temp.astype(np.uint8)
    #tahap1
    #Hashing Image yang akan menjadi key untuk block chipper
    for imageIndex in range(len(images)):
        temp = temp^(images[imageIndex])
    hash_method= cv2.img_hash.BlockMeanHash_create()
    output_hash=hash_method.compute(temp)

    return output_hash, temp


def multi_secret_image(images):
    images = [bit4to8(bit8to4(img)) for img in images]
    #(np.vectorize(float(secretImage[0])))
    output_hash, temp = hash_image(images)
    
    _,dim_hash=output_hash.shape
    height,width,channel=temp.shape

    R=np.zeros(temp.shape)
    #merubah temp menjadi grayscale
    temp_gs= 0.299 * temp[:,:,0] + 0.587 * temp[:,:,1] + 0.114 * temp[:,:,2]

    #Proses Block Chipper
    counter=0
    for i in range(height):
        for j in range(0,width,dim_hash):
            block_data=temp_gs[i,j:j+dim_hash]
            temp_mod= counter%dim_hash
            np.random.seed(block_data[temp_mod].astype('int64')+output_hash[0][temp_mod])
            block_data_random=np.random.randint(0,255,dim_hash)
            temp_gs[i][j:j+dim_hash]=block_data_random
            counter+=1

    R=np.zeros(images[0].shape).astype('uint8')
    R[:,:,0]=temp_gs
    R[:,:,1]=temp_gs
    R[:,:,2]=temp_gs
    
    
    randomImage=np.zeros((len(images),len(images[0]),len(images[0][0]),len(images[0][0][0]))).astype('uint8')
    numberOfSecretImage = len(images)
    #tahap 2
    for imageIndex in range(numberOfSecretImage):
        for x in range(height):
            for y in range(width):
                for z in range(channel):
                    randomImage[imageIndex][x][y][z]=(R[(x-imageIndex)%height][(y-imageIndex)%width][(z-imageIndex)%channel]^R[(x+imageIndex)%height][(y+imageIndex)%width][(z+imageIndex)%channel])
    #tahap 3
    sharedImage=[]
    for imageIndex in range(numberOfSecretImage-1):
        sharedImagePart=np.zeros(images[0].shape)
        sharedImagePart=images[imageIndex]^(randomImage[imageIndex]^randomImage[imageIndex+1])
        sharedImage.append(sharedImagePart)
    sharedImagePart=np.zeros(images[0].shape)
    sharedImagePart=images[imageIndex+1]^(randomImage[numberOfSecretImage-1]^randomImage[0])
    sharedImage.append(sharedImagePart)

    return sharedImage


def friendly_multi_secret_image_encrypt(multi_secret_images, mask):
    result = []
    for ms_img in multi_secret_images:
        bit_4_img = bit8to4(ms_img)
        cleanend_mask = bit4to8(bit8to4(mask))
        res = bit_4_img+cleanend_mask
        result.append(res)
    
    return result


def friendly_multi_secret_image_dencrypt(friendly_multi_secret_images):
    res = []
    for fms_img in friendly_multi_secret_images:
        subtract_img = bit4to8(bit8to4(fms_img))
        res.append(bit4to8(fms_img-subtract_img))
    
    return res


if __name__ == '__main__':
    image_path = ["image/1.tiff", "image/2.tiff", "image/3.tiff"]
    image_mask_path = "image/4.tiff"

    images = [cv2.imread(imp) for imp in image_path]
    mask = cv2.imread(image_mask_path)

    multi_secret_images = multi_secret_image(images)
    friendly_multi_secret_images = friendly_multi_secret_image_encrypt(multi_secret_images, mask)

    decode_friendly_multi_secret = friendly_multi_secret_image_dencrypt(friendly_multi_secret_images)
    decoded_img = multi_secret_image(decode_friendly_multi_secret)

    stacked_image = np.hstack(images)
    stacked_shared_image = np.hstack(multi_secret_images)
    stacked_friendly_image = np.hstack(friendly_multi_secret_images)
    stacked_decode_friendly = np.hstack(decode_friendly_multi_secret)
    stacked_decoded_image = np.hstack(decoded_img)

    print(psnr(stacked_decode_friendly, stacked_shared_image))
    print(stacked_decode_friendly)
    print(stacked_decoded_image)

    cv2.imwrite("assets/original_image.png", stacked_image)
    cv2.imwrite("assets/mask.png", mask)
    cv2.imwrite("assets/multi secret images.png", stacked_shared_image)
    cv2.imwrite("assets/friendly multi encrypted.png", stacked_friendly_image)
    cv2.imwrite("assets/decoded friendly multi encrypted.png", stacked_decode_friendly)
    cv2.imwrite("assets/decoded image.png", stacked_decoded_image)

    cv2.imshow("original_image", stacked_image)
    cv2.imshow("mask", mask)
    cv2.imshow("multi secret images", stacked_shared_image)
    cv2.imshow("friendly multi encrypted", stacked_friendly_image)
    cv2.imshow("decoded friendly multi encrypted", stacked_decode_friendly)
    cv2.imshow("decoded image", stacked_decoded_image)


    cv2.waitKey(0)