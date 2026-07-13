import torch

def avg_SNR(img,imgn):
    norm_img_sq = torch.norm(img, p='fro', dim=[2, 3]) ** 2
    norm_diff_sq = torch.norm(imgn - img, p='fro', dim=[2, 3]) ** 2
    snr = 10 * torch.log10(norm_img_sq / norm_diff_sq)
    average_snr = torch.mean(snr)
    return average_snr

def train(train_loader, model, criterion, optimizer, device):
    model.train()
    sum_loss, sum_snr, num = 0., 0., 0
    for (batch_x,label) in train_loader:
        batch_x = batch_x.to(device)
        label = label.to(device)

        max_vals = batch_x.abs().amax(dim=(1, 2, 3), keepdim=True)
        batch_x = batch_x / (max_vals + 1e-8)


        x_output = model(batch_x)
        loss = criterion(label / (max_vals + 1e-8), x_output)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_snr = avg_SNR(label, x_output * max_vals)
        sum_snr += avg_snr.item()
        sum_loss += loss.item()
        num += 1
    me_loss = sum_loss / num
    me_snr = sum_snr / num
    return me_loss, me_snr
