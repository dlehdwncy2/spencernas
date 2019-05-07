import  os , sys, cv2
from PIL import Image
import pefile
import binascii
import pandas as pd

def getBinaryData(filename):
    binaryValues = []
    file = open(filename, "rb")
    data = file.read(1)  # read byte by byte
    while data !=b"":
        try:
            binaryValues.append(chr(ord(data)))  # store value to array
        except TypeError:
            pass
        data = file.read(1)  # get next byte value

    return binaryValues

def createGreyScaleImageSpecificWith(dataSet,outputfilename,width=0):

    if (width == 0): # don't specified
        size = len(dataSet)

        if (size < 10240) :
            width = 32
        elif (10240 <= size <= 10240*3 ):
            width = 64
        elif (10240*3 <= size <= 10240*6 ):
            width = 128
        elif (10240*6 <= size <= 10240*10 ):
            width = 256
        elif (10240*10 <= size <= 10240*20 ):
            width = 384
        elif (10240*20 <= size <= 10240*50 ):
            width = 512
        elif (10240*50 <= size <= 10240*100 ):
            width = 768
        else :
            width = 1024

    height = int(size/width)+1

    image = Image.new('L', (width,height))

    image.putdata(dataSet)

    imagename = outputfilename+".bmp"
    image.save(imagename)
    image.show()
    print (imagename+" Greyscale image created")




class parshing_image():
    def __init__(self):
        if not (os.path.isdir("1_first_product")):
            os.makedirs("1_first_product")
        self.first_product_location = "1_first_product"       # [입력] 악성 코드의 디렉토리

        if not (os.path.isdir("2_second_product")):
            os.makedirs("2_second_product")
        self.second_product_location = "2_second_product"           # [출력] 첫번째 결과물 저장 장소 [ 악성코드 -> BIMAP ]

        if not (os.path.isdir("3_result_location")):
            os.makedirs("3_result_location")
        self.second_result_location = "3_result_location"          # [출력] 두번째 결과물 저장 장소 [ BITMAP -> 파싱 결과 ]


    def signature_confirm(self,dirname):
        file_list=os.listdir(dirname)
        mz_signature='MZ'
        pe_signature="PE"

        for file_name in file_list:
            full_file_name=os.path.join(dirname,file_name)
            try:
                #PE인지 아닌지 확인
                pe=pefile.PE(os.path.join(dirname,file_name))
                pe.close()
                continue
            except:
                buf=getBinaryData(full_file_name)
                print(buf[:30])
                print(buf[80:110])
                if mz_signature in buf[:30]:
                    if pe_signature in buf[80:110]: continue

                else: os.remove(os.path.join(dirname,file_name))


    def extratcion_bitmap(self, dirname):
        filenames = os.listdir(dirname)
        count = 0
        width = 100
        for filename in filenames:
            full_filename = os.path.join(dirname, filename)
            f_handle = open(full_filename, "rb")
            file_data = f_handle.read()
            size_data = 100-(len(file_data) % 100)
            if(size_data!=100):
                height = (len(file_data)+size_data)//100
            else:
                height = (len(file_data))//100
            f_handle.close()

            a_handle = open(full_filename,"ab")
            if(size_data!=100):
                for i in range(0,size_data):
                    a_handle.write('m'.encode())
            a_handle.close()

            r_handle=open(full_filename, "rb")
            data = r_handle.read()
            image = Image.frombytes('L', (width, height), data)

            image_save_full_path=os.path.join(self.second_product_location,filename+'.bmp')

            image.save(image_save_full_path)
            r_handle.close()
            count = count + 1
            os.remove(full_filename)

    def learning_image(self,dirname,index_num):
        count = 0
        Width = 256
        Height = 256
        filenames = os.listdir(dirname)

        image_save_full_path_list=[]
        for filename in filenames:
            full_filename = os.path.join(dirname, filename)
            if(full_filename.find('bmp')!=-1):
                count = count+1
                img = cv2.imread(full_filename, cv2.IMREAD_UNCHANGED)
                height, width = img.shape

                a = []
                [a.append(img[i,j]) for j in range(width) for  i in range(height)]


                b = [[0] * 256 for i in range(256)]

                for i in range(len(a) - 2):
                    b[a[i]][a[i + 1]] += 1

                c = sum(b, [])
                for i in range(len(c)):
                    if c[i] > 10:
                        c[i] = 255
                    else:
                        c[i] = 0
                image = Image.frombytes('L', (Width, Height), bytes(c))
                image_save_full_path=os.path.join(self.second_result_location,filename)
                image_save_full_path_list.append([image_save_full_path,index_num])
                image.save(image_save_full_path)
                os.remove(full_filename)

        return image_save_full_path_list

    def csv_save(self,image_save_full_path_list):
        pathfile="./group_index_csv"
        files_present=os.path.isfile(pathfile)
        dataframe = pd.DataFrame(image_save_full_path_list)

        if not files_present:
            dataframe.to_csv(pathfile,mode='w',header=False,index=False)
        else:
            dataframe.to_csv(pathfile,mode='a',header=False,index=False)

if __name__=="__main__":
    test=getBinaryData("D:\\Allinone\\BOB\\Python\\Tensflow\\TF\\convert_image\\1_first_product\\F8A22D446CB17EEF4E9855F92ED6D26ECDF8A0666A82116A986CC50733F0DD86")
    print(test[:150])

    print("Group Index Number : ")
    index_num=int(input())

    p_file=parshing_image()

    p_file.signature_confirm(p_file.first_product_location)  # EXE 파일 검증

    p_file.extratcion_bitmap(p_file.first_product_location) # bitmap extraction

    image_save_full_path_list=p_file.learning_image(p_file.second_product_location,index_num)     # bitmap_parshing


    p_file.csv_save(image_save_full_path_list)