# 保存模型示例代码
#在sharedeep的mnist做了实验
print('===> Saving models...')
state = {
    'state': model.state_dict(),
    'epoch': epoch                   # 将epoch一并保存
}
if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
torch.save(state, './checkpoint/autoencoder.t7')



#load_state_dict读取
print('===> Try resume from checkpoint')
if os.path.isdir('checkpoint'):
    try:
        checkpoint = torch.load('./checkpoint/autoencoder.t7')
        model.load_state_dict(checkpoint['state'])        # 从字典中依次读取
        start_epoch = checkpoint['epoch']
        print('===> Load last checkpoint data')
    except FileNotFoundError:
        print('Can\'t found autoencoder.t7')
else:
    start_epoch = 0
    print('===> Start from scratch')

	#https://www.cnblogs.com/qinduanyinghua/p/9311410.html
#字典


