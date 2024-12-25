import torch
from torch_ema import ExponentialMovingAverage
import math
import time
import numpy as np
from tqdm import tqdm
bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
textsize = 14
default_dtype = torch.float64

def loglinspace(rate, step, end=None):
    t = 0
    while end is None or t <= end:
        yield t
        t = int(t + 1 + step*(1 - math.exp(-t*rate/step)))

def evaluate(model, dataloader, loss_fn, loss_fn_mae, loss_fn_mse, device):
    model.eval()
    loss_cumulative = 0.
    rmse_cumulative_e = 0.
    rmse_cumulative_f = 0.
    mae_cumulative_e = 0.
    mae_cumulative_f = 0.
    # start_time = time.time()
    with torch.no_grad():
        for j, d in enumerate(dataloader):
            d.to(device)
            out_e, _, out_f = model(d)
            loss_e = loss_fn(out_e, d.energy)#.cpu()
            loss_f = loss_fn(out_f, d.force)#.cpu() #loss of force, the weight is for checking, should be change later
            loss_mae_e = loss_fn_mae(out_e, d.energy)#.cpu()
            loss_mae_f = loss_fn_mae(out_f, d.force)#.cpu()
            rmse_e = loss_fn_mse(out_e, d.energy)#.cpu()
            rmse_f = loss_fn_mse(out_f, d.force)#.cpu()
            rmse_cumulative_e = rmse_cumulative_e + rmse_e.detach().item()
            rmse_cumulative_f = rmse_cumulative_f + rmse_f.detach().item()
            mae_cumulative_e = mae_cumulative_e + loss_mae_e.detach().item()
            mae_cumulative_f = mae_cumulative_f + loss_mae_f.detach().item()
            loss_cumulative = loss_cumulative + loss_e.detach().item() + 36*loss_f.detach().item()
            #loss_cumulative_mae = loss_cumulative_mae + loss_mae_e.detach().item() + loss_mae_f.detach().item()
    return loss_cumulative / len(dataloader), mae_cumulative_e / len(dataloader), mae_cumulative_f / len(dataloader), np.sqrt(rmse_cumulative_e / len(dataloader)), np.sqrt(rmse_cumulative_f / len(dataloader))

def train(model, optimizer, dataloader_train, dataloader_valid, loss_fn, loss_fn_mae, loss_fn_mse, run_name,
          use_ema=True, ema_decay=float(0.99), ema_use_num_updates=True, max_epoch=101, scheduler=None, device="cpu"):
    model.to(device)
    best_metrics = float('inf')

    ema = None
    if use_ema and ema is None:
        ema = ExponentialMovingAverage(
                    model.parameters(),
                    decay=ema_decay,
                    use_num_updates=ema_use_num_updates,
                )

    #checkpoint_generator = loglinspace(0.3, 5)
    #checkpoint = next(checkpoint_generator)
    start_time = time.time()

#torch.save(model.state_dict(), '.pth')
    try: check_point = torch.load(run_name + '.torch')
    except:
        results = {}
        history = []
        start_epoch = 0
    else:
        model.load_state_dict(check_point['state'])
        history = check_point['history']
        start_epoch = history[-1]['epoch'] + 1
        optimizer.load_state_dict(check_point['optimizer_state_dict'])
        scheduler.load_state_dict(check_point['scheduler_state_dict'])


    for step in range(start_epoch, max_epoch):
        model.train()
        loss_cumulative = 0.
        loss_cumulative_mae = 0.
        
        for j, d in tqdm(enumerate(dataloader_train), total=len(dataloader_train), bar_format=bar_format):
            d.to(device)
            out_e, _, out_f = model(d)
            loss_e = loss_fn(out_e, d.energy)#.cpu()
            loss_f = loss_fn(out_f, d.force)#.cpu() #loss of force, the weight is for checking, should be change late
            loss_mae = loss_fn_mae(out_e, d.energy)#.cpu()
            loss = loss_e + 36*loss_f
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ema.update()

        end_time = time.time()
        wall = end_time - start_time

        #metrics
        train_avg_loss = evaluate(model, dataloader_train, loss_fn, loss_fn_mae, loss_fn_mse, device)

        if use_ema:
            with ema.average_parameters():
                valid_avg_loss = evaluate(model, dataloader_valid, loss_fn, loss_fn_mae, loss_fn_mse, device)
        else:
            valid_avg_loss = evaluate(model, dataloader_valid, loss_fn, loss_fn_mae, loss_fn_mse, device)    
        

        if valid_avg_loss[0] < best_metrics:
            best_metrics = valid_avg_loss[0]
            best_epoch = step + 1
            torch.save(model.state_dict(), 'best_model.pth')

        history.append({
            'epoch': step,
            'wall': wall,
            'batch': {
                'loss': loss.item(),
                'mean_abs': loss_mae.item(),
            },
            'valid': {
                'loss': valid_avg_loss[0],
                'mea_e': valid_avg_loss[1],
                'mea_f': valid_avg_loss[2],
                'rmse_e': valid_avg_loss[3],
                'rmse_f': valid_avg_loss[4],
            },
            'train': {
                'loss': train_avg_loss[0],
                'mea_e': train_avg_loss[1],
                'mea_f': train_avg_loss[2],
                'rmse_e': train_avg_loss[3],
                'rmse_f': train_avg_loss[4],
            },
        })

        results = {
            'history': history,
            'state': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        """
        print(f"Epoch {step+1:4d}   " + "\n" +
              "!!Train   " +
              f"loss = {train_avg_loss[0]:8.4f}   " +
              f"mae_e = {train_avg_loss[1]:8.4f}   " +
              f"mae_f = {train_avg_loss[2]:8.4f}   " +
              f"rmse_e = {train_avg_loss[3]:8.4f}   " +
              f"rmse_f = {train_avg_loss[4]:8.4f}   " + "\n" +
              "!!Val     " + 
              f"loss = {valid_avg_loss[0]:8.4f}   " +
              f"mae_e = {valid_avg_loss[1]:8.4f}   " +
              f"mae_f = {valid_avg_loss[2]:8.4f}   " +
              f"rmse_e = {valid_avg_loss[3]:8.4f}   " +
              f"rmse_f = {valid_avg_loss[4]:8.4f}   " + "\n" +
        print(f"elapsed time = {time.strftime('%H:%M:%S', time.gmtime(wall))}    " + f"Best epoch: {best_epoch:4d}")
        """
        with open(run_name + '.torch', 'wb') as f:
            torch.save(results, f)

        if scheduler is not None:
            scheduler.step(valid_avg_loss[0])
