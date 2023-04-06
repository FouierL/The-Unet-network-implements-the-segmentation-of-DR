import os
import numpy as np
import math
from packages.data_processing import trainGenerator, testGenerator, saveResult
from packages.model import build_model
from packages.evaluate import evl

if __name__ == '__main__':
    # 配置相关参数
    step = 'test'  # train or test
    log_name = 'IDRiD_exp'  # 本次训练的记录名称
    train_path = r'.\train'  # 训练集的路径
    img_folder_name = 'image'  #训练集图像文件夹名
    label_folder_name = 'label'  #训练集标签文件夹名
    test_path = r'.\test\image'  # 测试图像的路径
    groundtruth_path = r'.\test\label_binary'  # 测试图像真值的路径，如果没有label设置成None
    save_path = os.path.join(r'.\result', log_name)  # 结果保存路径
    load_dir = os.path.join(r'.\result', log_name,'weights.hdf5')  # 结果保存路径
    classes = 5  # 类别数量（包含背景类）
    target_size = (512, 512)  # 图像尺寸
    train_batch_size = 2  # 训练batch size，有BN层时最小为2
    test_batch_size = 1  # 测试batch size
    train_num = 54  # 训练样本数
    test_num = 27  # 测试样本数
    epochs = 40  # 最大迭代轮次
    learning_rate = 0.005  # 初始学习速率
    loss_f = 'diceloss'  # 选择损失函数，'diceloss'为Dice损失，‘CE‘为交叉熵损失
    save_result = True  # 是否保存可视化结果
    flag_multi_class = True

    os.makedirs(save_path, exist_ok=True)  # 新建结果和记录文件夹

    # 迭代器内置数据增强方法的参数配置
    train_args = dict(width_shift_range=20,  # 水平位移
                      height_shift_range=20,  # 垂直位移
                      vertical_flip=True,  # 垂直翻转
                      horizontal_flip=True,  # 水平翻转
                      fill_mode='constant')  # 空白区域填充模式

    # 定义训练迭代器
    trainGene = trainGenerator(train_batch_size,
                               train_args,
                               train_path,
                               img_folder_name,
                               label_folder_name,
                               shuffle=True,
                               flag_multi_class=flag_multi_class,
                               num_class=classes,
                               save_to_dir=None,
                               target_size=target_size)
    # 定义网络、优化器、损失函数
    model, opt, loss = build_model(classes=5,
                                   target_size=target_size,
                                   img_channel=3,
                                   learning_rate=0.0001,
                                   loss_f=loss_f)

    # 模型训练
    if step == 'train':
        model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])  # 编译模型，监视指标为acc
        train_names = ['train_loss', 'train_acc']
        x1 = trainGene
        loss_list = []
        for i in range(epochs):
            # 初始存放化训练记录的字典
            train_loss_on_epoch = {}
            val_loss_on_epoch = {}
            for k1 in train_names:
                train_loss_on_epoch[k1] = 0
            for j in range(math.ceil(train_num / train_batch_size)):
                train_x_y = next(x1)  # 从加载器中读取图像和标签
                train_x, train_y = train_x_y[0], train_x_y[1]
                loss_on_batch = model.train_on_batch(train_x, train_y)  # 使用train_on_batch方法训练模型，每次迭代返回为包含loss和监视指标的列表
                for m in range(len(loss_on_batch)):
                    train_loss_on_epoch[train_names[m]] = train_loss_on_epoch[train_names[m]] + loss_on_batch[m]
                print('ep:%d,%d/%d,loss:%f' % (i, j, math.ceil(train_num / train_batch_size), loss_on_batch[0]))
            for m in range(len(loss_on_batch)):  # 计算整个epoch的平均loss和acc
                train_loss_on_epoch[train_names[m]] = train_loss_on_epoch[train_names[m]] / train_num * train_batch_size
            print('ep:%d/%d' % (i, epochs), '\n', train_loss_on_epoch)
        model.save_weights(os.path.join(save_path, 'weights.hdf5'))  # 保存训练完毕的模型权重
        print('save ok')
    #模型测试
    elif step == 'test':  # 载入训练好的权重
        model.load_weights(load_dir, by_name=True)
        print('model loaded!')
    # 测试并可视化结果
    testGene = testGenerator(test_path, target_size=target_size)  # 定义测试集图片迭代器
    z1 = testGene
    results = np.empty((test_num, target_size[0], target_size[1], classes), np.float32)
    for n in range(math.ceil(test_num / test_batch_size)):
        test_x_y = next(z1)
        test_x = test_x_y
        print(test_x.shape)
        y_pred = model.predict_on_batch(test_x)
        print(y_pred.shape, y_pred.dtype)
        results[n] = np.squeeze(y_pred)
    # 保存分割结果
    if save_result == True:
        saveResult(save_path,
                   test_path,
                   target_size,
                   results,
                   classes=classes)
    if groundtruth_path:  #如果有测试数据集的label则进行模型性能评估
        # 计算评价指标并保存
        evl(results,
            classes=classes,
            target_size=(target_size[0], target_size[1]),
            groundtruth_path=groundtruth_path,
            save_path=save_path)
