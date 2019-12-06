def plot_reconstruction():
    print(" saving images at:{}".format(save_PATH))
    for indx in range(nSample):
    # Select images
        img = imgs[indx]
        img_variable = Variable(torch.FloatTensor(img))
        img_variable = img_variable.unsqueeze(0)
        img_variable = img_variable.to(device)
        imgs_z_mu, imgs_z_logvar = vae.Encoder(img_variable)
        imgs_z = vae.Reparam(imgs_z_mu, imgs_z_logvar)
        imgs_rec = vae.Decoder(imgs_z).cpu()
        imgs_rec = imgs_rec.data.numpy()
        img_i = imgs_rec[0] * 255.0
        img_i = img_i.transpose(1,0)
        # denormalize images
        img_i = img_i.reshape(height, width, 3)
        img_i = img_i.astype(np.uint8)
#        for ch in range(3):
#            img_i[:, :, ch] = img_i[:, :, ch] * imgs_std[ch] + imgs_mean[ch]
        io.imsave((save_PATH + '/background/imageRec%06d'%(indx+offset+1) + '.jpg'), img_i)
