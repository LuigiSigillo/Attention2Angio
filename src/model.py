def aagan(g_model_fine,g_model_coarse, d_model1, d_model2, d_model3, d_model4,image_shape_fine,image_shape_coarse,image_shape_x_coarse,label_shape_fine,label_shape_coarse):
    # Discriminator NOT trainable
    d_model1.trainable = False
    d_model2.trainable = False
    d_model3.trainable = False
    d_model4.trainable = False

    in_fine= Input(shape=image_shape_fine)
    in_coarse = Input(shape=image_shape_coarse)
    in_x_coarse = Input(shape=image_shape_x_coarse)
    label_fine = Input(shape=label_shape_fine)
    label_coarse = Input(shape=label_shape_coarse)

    # Generators
    gen_out_coarse, _ = g_model_coarse(in_coarse)
    gen_out_fine = g_model_fine([in_fine,in_x_coarse])

    # Discriminators Fine
    dis_out_1_fake = d_model1([in_fine, gen_out_fine])
    dis_out_2_fake = d_model2([in_fine, gen_out_fine])

    # Discriminators Coarse
    dis_out_3_fake = d_model3([in_coarse, gen_out_coarse])
    dis_out_4_fake = d_model4([in_coarse, gen_out_coarse])

    #feature matching loss

    fm1 = partial(feature_matching_loss, image_input=in_fine,real_samples=label_fine, D=d_model1)
    fm2 = partial(feature_matching_loss, image_input=in_fine,real_samples=label_fine, D=d_model2)
    fm3 = partial(feature_matching_loss, image_input=in_coarse,real_samples=label_coarse, D=d_model3)
    fm4 = partial(feature_matching_loss, image_input=in_coarse,real_samples=label_coarse, D=d_model4)

    model = Model([in_fine,in_coarse,in_x_coarse,label_fine,label_coarse], [dis_out_1_fake[0],
                                                    dis_out_2_fake[0],
                                                    dis_out_3_fake[0],
                                                    dis_out_4_fake[0],
                                                    gen_out_fine,
                                                    gen_out_fine,
                                                    gen_out_coarse,
                                                    gen_out_coarse,
                                                    gen_out_coarse,
                                                    gen_out_fine,
                                                    gen_out_coarse,
                                                    gen_out_fine,
                                                    gen_out_coarse,
                                                    gen_out_fine
                                                    ])

    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['hinge', 
                    'hinge',
                    'hinge',
                    'hinge',
                    fm1,
                    fm2,
                    fm3,
                    fm4,
                    'hinge',
                    'hinge',
                    'mse',
                    'mse',
                    perceptual_loss_coarse,
                    perceptual_loss_fine
                    ], 
              optimizer=opt,loss_weights=[1,1,1,1,
                                          1,
                                          1,
                                          1,
                                          1,
                                          10,10,10,10,10,10
                                          ])
    model.summary()
    return model
