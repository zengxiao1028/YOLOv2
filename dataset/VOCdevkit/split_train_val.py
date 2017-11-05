import os
from shutil import copyfile


def gen_voc(VOC_Dev_Folder, des_folder):

    split_file = os.path.join(VOC_Dev_Folder, 'ImageSets/Main/train.txt')
    if os.path.exists(split_file):
        with open(split_file,'r') as f:
            lines = [line.rstrip() for line in f]
            print(lines)


    for file in lines:
        annotation_file = os.path.join(os.path.join(VOC_Dev_Folder,'Annotations'), file + '.xml' )
        copyfile(annotation_file, os.path.join( os.path.join(des_folder,'train_ann'), file + '.xml'))

        annotation_file = os.path.join(os.path.join(VOC_Dev_Folder, 'JPEGImages'), file + '.jpg')
        copyfile(annotation_file, os.path.join(os.path.join(des_folder, 'train'), file + '.jpg'))

    split_file = os.path.join(VOC_Dev_Folder, 'ImageSets/Main/val.txt')
    if os.path.exists(split_file):
        with open(split_file, 'r') as f:
            lines = [line.rstrip() for line in f]
            print(lines)

    for file in lines:
        annotation_file = os.path.join(os.path.join(VOC_Dev_Folder, 'Annotations'), file + '.xml')
        copyfile(annotation_file, os.path.join(os.path.join(des_folder, 'val_ann'), file + '.xml'))

        annotation_file = os.path.join(os.path.join(VOC_Dev_Folder, 'JPEGImages'), file + '.jpg')
        copyfile(annotation_file, os.path.join(os.path.join(des_folder, 'val'), file + '.jpg'))



if __name__ == '__main__':
    gen_voc('./VOC2007',
            '../voc2007')


