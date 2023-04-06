
import cv2
import os
import numpy as np


def evl(results,
        classes=5,
        target_size=(512, 512),
        groundtruth_path=None,
        save_path=None):

    def get_y_true(indir, classes, target_size=(512, 512)):  # get y_true from groundthruth
        class1_indir = os.path.join(indir, 'EX')
        class2_indir = os.path.join(indir, 'HE')
        class3_indir = os.path.join(indir, 'MA')
        class4_indir = os.path.join(indir, 'SE')
        filelist = os.listdir(class1_indir)
        classes_np_list = []
        h, w = target_size
        if classes == 1:
            classes_np = np.empty((classes,) + (len(filelist),) + target_size, )
        else:
            classes_np = np.empty((classes - 1,) + (len(filelist),) + target_size, )
        for j, filename in enumerate(filelist):
            label1 = cv2.resize(cv2.imread(os.path.join(class1_indir, filename), 0), target_size, cv2.INTER_NEAREST)
            label2 = cv2.resize(cv2.imread(os.path.join(class2_indir, filename), 0), target_size, cv2.INTER_NEAREST)
            label3 = cv2.resize( cv2.imread(os.path.join(class3_indir, filename), 0), target_size, cv2.INTER_NEAREST)
            label4 = cv2.resize(cv2.imread(os.path.join(class4_indir, filename), 0), target_size, cv2.INTER_NEAREST)
            label_1 = np.zeros(target_size)
            label_2 = np.zeros(target_size)
            label_3 = np.zeros(target_size)
            label_4 = np.zeros(target_size)
            label_1[np.where(label1 == 255)] = 1
            label_2[np.where(label2 == 255)] = 1
            label_3[np.where(label3 == 255)] = 1
            label_4[np.where(label4 == 255)] = 1
            classes_np[(0, j)] = label_1
            classes_np[(1, j)] = label_2
            classes_np[(2, j)] = label_3
            classes_np[(3, j)] = label_4
            classes_np1 = classes_np[0].reshape(len(filelist) * h * w)
            classes_np_list.append(classes_np1)
            classes_np2 = classes_np[1].reshape(len(filelist) * h * w)
            classes_np_list.append(classes_np2)
            classes_np3 = classes_np[2].reshape(len(filelist) * h * w)
            classes_np_list.append(classes_np3)
            classes_np4 = classes_np[3].reshape(len(filelist) * h * w)
            classes_np_list.append(classes_np4)
        print('classes_np', classes_np.shape)
        print('classes_np1', classes_np1.shape)
        return classes_np_list

    def get_y_pred(results, classes, target_size=(512, 512)):  # get y_pred from predict-results
        y_pred = []
        h, w = target_size
        results_flatten = results.reshape((len(results) * w * h, classes))
        if classes == 1:
            y_pred.append(results_flatten[:, 0])
        else:
            for i in range(classes):
                y_pred.append(results_flatten[:, i])  # y_pred[0] is background class
        return y_pred
    y_true = get_y_true(groundtruth_path, classes, target_size=target_size)  # transforming every image into array and saving in a list
    y_pred = get_y_pred(results, classes, target_size=target_size)

    """
    evaluation methods

    """

    def compute_pr_f1(y_pred, y_true, class_num=3):
        f1_dict = {}

        if class_num == 5:
            class_list = ['EX', 'HE', 'MA', 'SE']
            print(np.array(y_pred).shape)
            mf1 = 0
            for j, class_name in enumerate(class_list):
                if len(y_pred[j + 1]) != len(y_true[j]):
                    raise ValueError("Shape mismatch: y_pred and y_true must have the same shape.")
                pos_prob = np.zeros(len(y_pred[0]))
                for i in range(len(y_pred[0])):
                    pred_1 = y_pred[0][i]
                    pred_2 = y_pred[1][i]
                    pred_3 = y_pred[2][i]
                    pred_4 = y_pred[3][i]
                    pred_5 = y_pred[4][i]
                    class_arr = np.array([pred_1, pred_2, pred_3, pred_4, pred_5])
                    item = np.argmax(class_arr)
                    if item == j + 1:  # pred_1 is background
                        pos_prob[i] = 1
                pos_prob = pos_prob.astype(np.bool)
                pos_true = np.array(y_true[j]).astype(np.bool)
                tp = np.logical_and(pos_prob, pos_true).sum()
                true_pos = pos_true.sum()
                prob_pos = pos_prob.sum()
                fp = prob_pos - tp
                tn = len(y_pred[0]) - true_pos - prob_pos + tp
                recall = tp / true_pos
                precision = tp / prob_pos
                sp = tn / (tn + fp)
                f1 = 2 * recall * precision / (recall + precision)
                mf1 = mf1 + f1
                f1_dict.setdefault('class_name', []).append(class_name)
                f1_dict.setdefault('recall', []).append(recall)
                f1_dict.setdefault('precision', []).append(precision)
                f1_dict.setdefault('F1-score', []).append(f1)
                f1_dict.setdefault('specificity ', []).append(sp)
                print(class_name + ' : ', 'recall = ', recall, ' precision = ', precision, ' F1-score= ', f1)
            f1_dict['mean_F1'] = mf1 / len(class_list)
            print('mean_F1:%f' % (mf1 / len(class_list)))
        return f1_dict

    f1_dict = compute_pr_f1(y_pred, y_true, class_num=classes)  # 计算pr,recall,f1
    #保存结果到txt文件
    f = open(os.path.join(save_path,'f1.txt'), 'w', encoding='utf-8')  # 以'w'方式打开文件
    for k, v in f1_dict.items():  # 遍历字典中的键值
        s2 = str(v)  # 把字典的值转换成字符型
        f.write(k + '\n')  # 键和值分行放，键在单数行，值在双数行
        f.write(s2 + '\n')
    f.close()  # 关闭文件