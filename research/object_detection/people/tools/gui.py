# -*- coding:utf-8 -*-
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
from tkinter import ttk
import os

from people.eval.predict_images import predict_image


class TKS(tk.Tk):
    def __init__(self):
        super(TKS, self).__init__()

        self.flag = 0

        self.content_of_pic = ''

        self.img = ''
        self.filename = ''
        self.filename_ori = ''

        self.pic_full_path = ''
        self.dir_path = ''

        self.list_pic_dir = []

        self.index = 0

        self.entryvar_wrong_num = tk.StringVar()
        self.entryvar_all_num = tk.StringVar()
        self.entryvar_all_score = tk.StringVar()

        self.frame_content = tk.LabelFrame(self, text='content', height=40, width=800)
        self.content_entry = tk.Entry(self.frame_content, width = 50)
        self.content_entry.pack(side='left')

        self.content_button = tk.Button(self.frame_content, height=1, width=5, text='目录', command=self.get_dir)
        self.content_button.pack(side='left')
        self.frame_content.pack()

        self.top_frame = tk.LabelFrame(self, text='left', height=40, width=800)

        self.content_entry_pic_name = tk.Entry(self.top_frame, width=50)
        self.content_entry_pic_name.pack(side='left')

        self.content_button_pic_name = tk.Button(self.top_frame, width=5, height=1, text='下一页', command=self.get_pic_path)
        self.content_button_pic_name.pack(side='left')

        self.top_frame.pack()

        self.frame_pic_all = tk.LabelFrame(self, text='pic_left_right')

        self.frame_pic = tk.LabelFrame(self.frame_pic_all, text='pic')

        self.frame_pic_Scrollbar_col = ttk.Scrollbar(self.frame_pic, orient=tk.VERTICAL)
        self.frame_pic_Scrollbar_col.pack(side=tk.RIGHT, fill=tk.Y)

        self.frame_pic_Scrollbar_row = ttk.Scrollbar(self.frame_pic, orient=tk.HORIZONTAL)
        self.frame_pic_Scrollbar_row.pack(side=tk.BOTTOM, fill=tk.X)

        self.pic_canvas = tk.Canvas(self.frame_pic, bg='white', height=800, width=800, scrollregion=(0,0,1600,1600))
        self.pic_canvas.pack(side=tk.LEFT)

        self.frame_pic_Scrollbar_col.config(command=self.pic_canvas.yview)
        self.frame_pic_Scrollbar_row.config(command=self.pic_canvas.xview)
        self.pic_canvas['xscrollcommand']=self.frame_pic_Scrollbar_row.set
        self.pic_canvas['yscrollcommand'] = self.frame_pic_Scrollbar_col.set
        self.frame_pic_Scrollbar_col['command']=self.pic_canvas.yview
        self.frame_pic_Scrollbar_row['command']=self.pic_canvas.xview
        self.frame_pic.pack(side='left')

        self.frame_pic_right = tk.LabelFrame(self.frame_pic_all, text='pic_right')

        self.frame_pic_Scrollbar_col_right = ttk.Scrollbar(self.frame_pic_right, orient=tk.VERTICAL)
        self.frame_pic_Scrollbar_col_right.pack(side=tk.RIGHT, fill=tk.Y)

        self.frame_pic_Scrollbar_row_right = ttk.Scrollbar(self.frame_pic_right, orient=tk.HORIZONTAL)
        self.frame_pic_Scrollbar_row_right.pack(side=tk.BOTTOM, fill=tk.X)

        self.pic_canvas_right = tk.Canvas(self.frame_pic_right, bg='white', height=800, width=800, scrollregion=(0, 0, 1600, 1600))
        self.pic_canvas_right.pack(side=tk.LEFT)

        self.frame_pic_Scrollbar_col_right.config(command=self.pic_canvas_right.yview)
        self.frame_pic_Scrollbar_row_right.config(command=self.pic_canvas_right.xview)
        self.pic_canvas_right['xscrollcommand'] = self.frame_pic_Scrollbar_row_right.set
        self.pic_canvas_right['yscrollcommand'] = self.frame_pic_Scrollbar_col_right.set
        self.frame_pic_Scrollbar_col_right['command'] = self.pic_canvas_right.yview
        self.frame_pic_Scrollbar_row_right['command'] = self.pic_canvas_right.xview
        self.frame_pic_right.pack()
        self.frame_pic_all.pack()

    def get_dir(self):
        self.entryvar_dir = tk.StringVar()
        self.dir_path = filedialog.askdirectory()

        self.entryvar_dir.set(self.dir_path)
        self.content_entry.config(textvariable=self.entryvar_dir)
        ex_list = os.listdir(self.dir_path)
        ex_list.sort()
        self.list_pic_dir = ex_list
        self.entryvar_pic_path = tk.StringVar()
        self.entryvar_pic_path.set(self.list_pic_dir[self.index])
        self.content_entry_pic_name.config(textvariable=self.entryvar_pic_path)

        self.pic_full_path = self.dir_path + '/' + self.list_pic_dir[self.index]
        self.img = Image.open(self.pic_full_path)
        print(type(self.img))
        self.filename_ori = ImageTk.PhotoImage(self.img)
        self.pic_canvas.create_image(self.img.size[0]/2, self.img.size[1]/2, image=self.filename_ori)

        img = pri.pri_pic(self.pic_full_path)
        self.filename = ImageTk.PhotoImage(img)
        self.pic_canvas_right.create_image(self.img.size[0]/2, self.img.size[1]/2, image=self.filename)

        # print('path is:', self.pic_full_path)


    def get_pic_path(self):
        self.flag = 1
        self.index += 1
        self.entryvar_pic_path = tk.StringVar()
        try:
            self.entryvar_pic_path.set(self.list_pic_dir[self.index])
            self.content_entry_pic_name.config(textvariable=self.entryvar_pic_path)
            self.pic_full_path = self.dir_path+'/'+self.list_pic_dir[self.index]
            if self.pic_full_path.split('.')[-1] == 'jpg' or self.pic_full_path.split('.')[-1] == 'png':
                # self.pic_full_path = '/home/baobao/Documents/11_28/models-master/research/object_detection/people_count/13-15-32-1130-img-17.jpg'
                self.img = Image.open(self.pic_full_path)
                self.filename_ori = ImageTk.PhotoImage(self.img)
                img = predict_image(' graph.pd', '.pdtxt ', [self.pic_full_path])[0]
                self.filename = ImageTk.PhotoImage(img)
                print(type(self.filename))
                self.pic_canvas.create_image(self.img.size[0] / 2, self.img.size[1] / 2, image=self.filename_ori)
                self.pic_canvas_right.create_image(self.img.size[0] / 2, self.img.size[1] / 2, image=self.filename)
                print('path is:', self.pic_full_path)

            else:
                print('the path is not pic(jpg or png):', self.pic_full_path)


        except BaseException:
            print("pa in except", self.pic_full_path)
            messagebox.showerror('Python Tkinter', "最后一页")


if __name__ == '__main__':
    app = TKS()
    app.mainloop()