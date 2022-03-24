import os
import torch
from tensorboardX import SummaryWriter
from data import DataScheduler


def train_model(config, model, scheduler: DataScheduler, writer: SummaryWriter, unlabeled_loader, noisy_loader, origin):
    saved_model_path = os.path.join(config['log_dir'], 'ckpts')
    os.makedirs(saved_model_path, exist_ok=True) 
    prev_t = 0
    '''
    for step, ((x, y, corrupt, idx), t) in enumerate(scheduler):
        # Evaluate the model when task changes
        if t != prev_t:
            scheduler.eval(model, writer, step + 1, eval_title='eval')
        # learn the model
        for i in range(config['batch_iter']):
            # x는 image, y는 label, corrupt는 올바른 것인지 noisy한 것인지를 의미
            model.learn(x, y, corrupt, idx, step * config['batch_iter'] + i)

        prev_t = t

    # evlauate the model when all of takss are trained
    scheduler.eval(model, writer, step + 1, eval_title='eval')
    torch.save(model.state_dict(), os.path.join(
        saved_model_path, 'ckpt-{}'.format(str(step + 1).zfill(6))))
    writer.flush()
    '''
    print("-------train SPR------")
    for idx, data in enumerate(noisy_loader):
        image_names = data['image_name']
        labels = data['label']
        corrupted = []
        files = []
        for i in range(len(image_names)):
            #print("origin",origin[image_names[i]].item(), 'label', labels[i].item())
            #print(origin[image_names[i]].item()!=labels[i].item())
            corrupted.append(origin[image_names[i]].item()!=labels[i].item())
            image_name = image_names[i].split('/')[2].split('.')[0]
            files.append(int(image_name))
        data['corrupt'] = torch.Tensor(corrupted)
        #print("corrupt!!")
        #print(data['corrupt'])
        data['image_names'] = files
        model.learn(data['image'], labels, data['corrupt'], torch.Tensor(files), idx)


    print("-------inference SPR------")
    #scheduler.eval(model, writer, step + 1, eval_title='eval')
    #eval model과 noisy loader이용
    purified = model.inference(noisy_loader)
    print("purified", purified)
    torch.save(model.state_dict(), os.path.join(saved_model_path, 'ckpt-model'))
    writer.flush()
   
    #return purified
