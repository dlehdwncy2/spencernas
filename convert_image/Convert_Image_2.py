import  os , sys, cv2
from PIL import Image
import pefile
import binascii
import pandas as pd


class convert_binary_to_image:
    def __init__(self):
        if not (os.path.isdir("converter_directory")):os.makedirs("converter_directory")
        self.first_product_location = "converter_directory"


    def binary_value_get(self,filename):
        Bin_value= []
        with open(filename,'rb') as file_object:
            data=file_object.read(1)
            while data !=b"":
                try:
                    Bin_value.append(chr(ord(data)))
                except TypeError:
                    pass
                data=file_object.read(1)
            return Bin_value

    def createGreyScaleImageSpecificWith(self,dataSet,outputfilename,width=0):

        if (width == 0):
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
        imagename = outputfilename
        image.save(imagename)
        #image.show()

    def signature_confirm(self,dirname):
        file_list=os.listdir(dirname)
        mz_signature='MZ'
        pe_signature="PE"

        for file_name in file_list:
            full_file_name=os.path.join(dirname,file_name)
            try:
                pe=pefile.PE(os.path.join(dirname,file_name))
                pe.close()
                continue
            except:
                buf=self.binary_value_get(full_file_name)
                if mz_signature in buf[:30]:
                    if pe_signature in buf[80:110]: continue
                else: os.remove(os.path.join(dirname,file_name))

    def extratcion_bitmap(self, dirname,index_num):
        filenames = os.listdir(dirname)
        extraction_width = 100

        #학습 이미지 크기
        Lean_Width = 256
        Lean_Height = 256

        image_save_full_path_list=[]
        for filename in filenames:
            full_filename = os.path.join(dirname, filename)
            f_handle = open(full_filename, "rb")
            file_data = f_handle.read()
            size_data = 100-(len(file_data) % 100)
            if(size_data!=100):
                Extraction_height = (len(file_data)+size_data)//100
            else:
                Extraction_height = (len(file_data))//100
            f_handle.close()

            a_handle = open(full_filename,"ab")
            if(size_data!=100): [a_handle.write('m'.encode()) for _ in range(0,size_data)]
            a_handle.close()

            r_handle=open(full_filename, "rb")
            data = r_handle.read()
            image = Image.frombytes('L', (extraction_width, Extraction_height ), data)

            image_save_full_path=os.path.join(dirname,filename+'.bmp')

            image.save(image_save_full_path)
            r_handle.close()
            os.remove(full_filename)

            Lean_img= cv2.imread(image_save_full_path, cv2.IMREAD_UNCHANGED)

            img_height, img_width = Lean_img.shape

            a = []
            [a.append(Lean_img[i, j]) for j in range(img_width) for i in range(img_height)]
            b = [[0] * 256 for i in range(256)]
            for i in range(len(a) - 2): b[a[i]][a[i + 1]] += 1
            c = sum(b, [])

            for i in range(len(c)):
                if c[i] > 10:c[i] = 255
                else:c[i] = 0
            image = Image.frombytes('L', (Lean_Width, Lean_Height), bytes(c))
            image_save_full_path_list.append([image_save_full_path, index_num])
            image.save(image_save_full_path)

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
    p_file=convert_binary_to_image()
    print("Group Index Number : ")
    index_num=int(input())

    p_file.signature_confirm(p_file.first_product_location)  # EXE 파일 검증

    image_save_full_path_list=p_file.extratcion_bitmap(p_file.first_product_location,index_num) # bitmap extraction


    p_file.csv_save(image_save_full_path_list)