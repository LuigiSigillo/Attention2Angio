from src.model import coarse_generator,fine_generator,aagan,discriminator
from src.performance_visualize import visualize_save_weight, visualize_save_weight_global, plot_history
from src.dataloader import resize, generate_fake_data_coarse, generate_fake_data_fine, generate_real_data, load_real_data
import argparse
import time
from numpy import load
import gc
import keras.backend as K
from functools import partial

def train(d_model1, d_model2, d_model3, d_model4, g_global_model, g_local_model, 
          gan_model, dataset, n_epochs=20, n_batch=1, n_patch=[64,32],savedir='AAGAN'):
    
    if not os.path.exists(savedir):
      os.makedirs(savedir)
    # unpack dataset
    trainA, trainB = dataset
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    
    # lists for storing loss, for plotting later
    d1_hist, d2_hist, d3_hist, d4_hist, d5_hist, d6_hist, d7_hist, d8_hist =  list(),list(), list(), list(), list(),list(), list(), list()
    fm1_hist,fm2_hist, fm3_hist,fm4_hist= list(),list(), list(), list()
    g_global_hist, g_local_hist, gan_hist =  list(), list(), list()
    g_global_percp_hist, g_local_percp_hist, g_global_recon_hist, g_local_recon_hist =list(),list(), list(), list()
    # manually enumerate epochs
    for i in range(n_steps):
        d_model1.trainable = True
        d_model2.trainable = True
        d_model3.trainable = True
        d_model4.trainable = True
        gan_model.trainable = False
        g_global_model.trainable = False
        g_local_model.trainable = False
        for j in range(2):
            # select a batch of real samples 
            [X_realA, X_realB], [y1,y2,y3] = generate_real_data(dataset, n_batch, n_patch)

            
            # generate a batch of fake samples for Coarse Generator
            out_shape = (int(X_realA.shape[1]/2),int(X_realA.shape[2]/2))
            [X_realA_half,X_realB_half] = resize(X_realA,X_realB,out_shape)
            [X_fakeB_half, x_global], [y1_coarse,y2_coarse] = generate_fake_data_coarse(g_global_model, X_realA_half, n_patch)


            # generate a batch of fake samples for Fine Generator
            X_fakeB, [y1_fine,y2_fine] = generate_fake_data_fine(g_local_model, X_realA, x_global, n_patch)


            ## FINE DISCRIMINATOR  
            # update discriminator for real samples
            d_loss1 = d_model1.train_on_batch([X_realA, X_realB], y1)[0]
            # update discriminator for generated samples
            d_loss2 = d_model1.train_on_batch([X_realA, X_fakeB], y1_fine)[0]

            #d_loss1 = 0.5*(d_loss1_real[0]+d_loss1_fake[0])

            # update discriminator for real samples
            d_loss3 = d_model2.train_on_batch([X_realA, X_realB], y2)[0]
            # update discriminator for generated samples
            d_loss4 = d_model2.train_on_batch([X_realA, X_fakeB], y2_fine)[0]
            
            #d_loss2 = 0.5*(d_loss2_real[0]+d_loss2_fake[0])

            ## COARSE DISCRIMINATOR  
            # update discriminator for real samples
            d_loss5 = d_model3.train_on_batch([X_realA_half, X_realB_half], y2)[0]
            # update discriminator for generated samples
            d_loss6 = d_model3.train_on_batch([X_realA_half, X_fakeB_half], y1_coarse)[0]

            #d_loss3 = 0.5*(d_loss3_real[0]+d_loss3_fake[0])

            # update discriminator for real samples
            d_loss7 = d_model4.train_on_batch([X_realA_half, X_realB_half], y3)[0]
            # update discriminator for generated samples
            d_loss8 = d_model4.train_on_batch([X_realA_half, X_fakeB_half], y2_coarse)[0]
            
            #d_loss4 = 0.5*(d_loss4_real[0]+d_loss4_fake[0])
        
        #if n_steps%425 ==0:

        # turn Global G1 trainable
        d_model1.trainable = False
        d_model2.trainable = False
        d_model3.trainable = False
        d_model4.trainable = False
        gan_model.trainable = False
        g_global_model.trainable = True
        g_local_model.trainable = False
        
        

        # select a batch of real samples for Local enhancer
        [X_realA, X_realB], _ = generate_real_data(dataset, n_batch, n_patch)

        # Global Generator image fake and real
        out_shape = (int(X_realA.shape[1]/2),int(X_realA.shape[2]/2))
        [X_realA_half,X_realB_half] = resize(X_realA,X_realB,out_shape)
        [X_fakeB_half, x_global], _ = generate_fake_data_coarse(g_global_model, X_realA_half, n_patch)
        

        # update the global generator
        g_global_loss,_ = g_global_model.train_on_batch(X_realA_half, [X_realB_half])

        
        d_model1.trainable = False
        d_model2.trainable = False
        d_model3.trainable = False
        d_model4.trainable = False
        gan_model.trainable = False
        g_global_model.trainable = False
        g_local_model.trainable = True
        
        # update the Local Enhancer 
        g_local_loss = g_local_model.train_on_batch([X_realA,x_global], X_realB)
        

        # turn G1, G2 and GAN trainable, not D1,D2 and D3
        d_model1.trainable = False
        d_model2.trainable = False
        d_model3.trainable = False
        d_model4.trainable = False
        gan_model.trainable = True
        g_global_model.trainable = True
        g_local_model.trainable = True
        # update the generator

        gan_loss,_,_,_,_,fm1_loss,fm2_loss,fm3_loss,fm4_loss,_,_,g_global_recon_loss, g_local_recon_loss, g_global_percp_loss, g_local_percp_loss = gan_model.train_on_batch([X_realA,X_realA_half,x_global,
                                                                                                                                                     X_realB,X_realB_half], 
                                                                                                                                                    [y1, y2, y2, y3,
                                                                                                                                                     X_fakeB,X_fakeB,X_fakeB_half,X_fakeB_half,
                                                                                                                                                     X_realB_half,X_realB,
                                                                                                                                                     X_realB_half,X_realB,
                                                                                                                                                     X_realB_half,X_realB
                                                                                                                                         ])

        # summarize performance
        print('>%d, d1[%.3f] d2[%.3f] d3[%.3f] d4[%.3f] d5[%.3f] d6[%.3f] d7[%.3f] d8[%.3f] fm1[%.3f] fm2[%.3f] fm3[%.3f] fm4[%.3f] g_g[%.3f] g_l[%.3f] g_g_r[%.3f] g_l_r[%.3f] g_g_p[%.3f] g_l_p[%.3f] gan[%.3f]' % 
              (i+1, d_loss1, d_loss2, d_loss3, d_loss4, d_loss5, d_loss6, d_loss7, d_loss8, 
               fm1_loss, fm2_loss, fm3_loss, fm4_loss, 
               g_global_loss, g_local_loss, 
               g_global_recon_loss, g_local_recon_loss, g_global_percp_loss, g_local_percp_loss, gan_loss))
                                                                                                                           
        d1_hist.append(d_loss1)
        d2_hist.append(d_loss2)
        d3_hist.append(d_loss3)
        d4_hist.append(d_loss4)
        d5_hist.append(d_loss5)
        d6_hist.append(d_loss6)
        d7_hist.append(d_loss7)
        d8_hist.append(d_loss8)
        fm1_hist.append(fm1_loss)
        fm2_hist.append(fm2_loss)
        fm3_hist.append(fm3_loss)
        fm4_hist.append(fm4_loss)
        g_global_hist.append(g_global_loss)
        g_local_hist.append(g_local_loss)
        g_global_recon_hist.append(g_global_recon_loss)
        g_local_recon_hist.append(g_local_recon_loss)
        g_global_percp_hist.append(g_global_percp_loss)
        g_global_percp_hist.append(g_local_percp_loss)
        gan_hist.append(gan_loss)
        # summarize model performance
        if (i+1) % (bat_per_epo * 1) == 0:
            summarize_performance_global(i, g_global_model, dataset, n_samples=3,savedir=savedir)
            
            summarize_performance(i, g_global_model,g_local_model, dataset, n_samples=3,savedir=savedir)
    plot_history(d1_hist, d2_hist, d3_hist, d4_hist, d5_hist, d6_hist, d7_hist, d8_hist, fm1_hist, fm2_hist, fm3_hist, fm4_hist, g_global_hist,g_local_hist, 
                 g_global_percp_hist, g_local_percp_hist, g_global_recon_hist, g_local_recon_hist, gan_hist,savedir=savedir)
    to_csv(d1_hist, d2_hist, d3_hist, d4_hist, d5_hist, d6_hist, d7_hist, d8_hist, fm1_hist, fm2_hist, fm3_hist, fm4_hist, g_global_hist,g_local_hist, 
                 g_global_percp_hist, g_local_percp_hist, g_global_recon_hist, g_local_recon_hist, gan_hist,savedir=savedir)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--npz_file', type=str, default='attenton2angio', help='path/to/npz/file')
    parser.add_argument('--input_dim', type=int, default=512)
    parser.add_argument('--savedir', type=str, required=False, help='path/to/save_directory',default='AAGAN')
    args = parser.parse_args()

    K.clear_session()
    gc.collect()
    start_time = time.time()
    dataset = load_real_data(args.npz_file+'.npz')
    print('Loaded', dataset[0].shape, dataset[1].shape)
    
    # define input shape based on the loaded dataset
    in_size = args.input_dim
    image_shape_coarse = (in_size//2,in_size//2,3)
    label_shape_coarse = (in_size//2,in_size//2,1)

    #image_shape_coarse2 = (in_size/4,in_size/4,3)
    #label_shape_coarse2 = (in_size/4,in_size/4,1)

    image_shape_fine = (in_size,in_size,3)
    label_shape_fine = (in_size,in_size,1)

    image_shape_xglobal = (in_size//2,in_size//2,64)
    ndf=32
    ncf=64
    nff=64
    # define discriminator models
    d_model1 = discriminator(image_shape_fine,label_shape_fine,ndf,n_downsampling=0,name="D1") # D1 Fine
    d_model2 = discriminator(image_shape_fine,label_shape_fine,ndf,n_downsampling=1,name="D2") # D2 Fine 

    d_model3 = discriminator(image_shape_coarse,label_shape_coarse,ndf,n_downsampling=0,name="D3") # D1 Coarse
    d_model4 = discriminator(image_shape_coarse,label_shape_coarse,ndf,n_downsampling=1,name="D4") # D2 Coarse


    # define generator models
    g_coarse_model = coarse_generator(img_shape=image_shape_coarse,n_downsampling=2, n_blocks=9, n_channels=1)

    g_fine_model = fine_generator(x_coarse_shape=image_shape_xglobal,input_shape=image_shape_fine,nff=nff,n_blocks=3)

    # define fundus2angio 
    gan_model = aagan(g_fine_model,g_coarse_model, d_model1, d_model2, d_model3, d_model4,
                  image_shape_fine,image_shape_coarse, image_shape_xglobal,label_shape_fine,label_shape_coarse)
    # train model
    train(d_model1, d_model2, d_model3, d_model4,g_coarse_model, g_fine_model, gan_model, dataset, n_epochs=args.epochs, n_batch=args.batch_size, n_patch=[64,32,16],savedir=args.savedir)
  
    end_time = time.time()
    time_taken = (end_time-start_time)/3600.0
    print(time_taken)
