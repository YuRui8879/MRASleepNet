import os
import time

class Log:
    def __init__(self,save_path):
        time_str = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        file_name = os.path.join(save_path,time_str)
        self.f = open(file_name,'w+')
        print('open file success')

    def log(self,txt):
        print(txt)
        self.f.write(txt)
        self.f.write('\n')

    def close(self):
        self.f.close()
        print('close file success')