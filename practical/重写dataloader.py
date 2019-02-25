#参考 ： https://www.cnblogs.com/kk17/p/10105862.html



data_list=['1','2','3','4']
label_list=['0','1','0','0']








train_data  = trainset()#实例化dataset
#常规做法用torchvision.datasets.ImageFolder读路径
#但如果我的图片格式不是jpg,img这种，那怎么办呢？
trainloader = DataLoader(train_data, batch_size=4,shuffle=True)#然后用dataloader包起来