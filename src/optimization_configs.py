
all_lines_kitti_auto= [
    f'model_optimization_flavor(optimization_level=2, compression_level=0, batch_size=1)',
    f'model_optimization_config(calibration, calibset_size={cs_size}, batch_size=1)',
    'post_quantization_optimization(finetune, policy=disabled)',
    'post_quantization_optimization(bias_correction,  policy=enabled, batch_size=1)',
]


all_lines_kitti_1= [
    f'model_optimization_flavor(optimization_level=2, compression_level=0, batch_size=1)',
    'model_optimization_config(checker_cfg, batch_size=1)',
    f'model_optimization_config(calibration, calibset_size={cs_size}, batch_size=1)',
    # 'pre_quantization_optimization(activation_clipping, layers=[conv1,conv2,conv4,conv5], mode=percentile, clipping_values=[0.01, 99.999])',
    # 'pre_quantization_optimization(activation_clipping, layers=[conv3,conv6,conv7,conv8,conv9,conv10,conv11,conv12,deconv1,conv17,deconv2], mode=percentile, clipping_values=[0.01, 99.995])',
    # 'pre_quantization_optimization(activation_clipping, layers=[conv13,conv14,conv15,conv16], mode=percentile, clipping_values=[0.01, 99.95])',
    # 'pre_quantization_optimization(activation_clipping, layers=[conv18], mode=percentile, clipping_values=[0.01, 99.99])',    
    # 'pre_quantization_optimization(activation_clipping, layers=[conv19], mode=percentile, clipping_values=[0.01, 99.999])',
    #'pre_quantization_optimization(activation_clipping, layers=[conv20], mode=percentile, clipping_values=[0.001, 99.999])',
    'pre_quantization_optimization(activation_clipping, layers={*}, mode=percentile, clipping_values=[0.001, 99.99])',
    #'quantization_param({conv*}, null_channels_cutoff_factor=1e2)',
    #clipping_values=[0.001, 99.99])', hasta ahora el mejor
   
    'post_quantization_optimization(finetune, policy=enabled)',
    'post_quantization_optimization(bias_correction,  policy=enabled, batch_size=1)',
    
    #'resources_param(max_utilization=0.8)'

]

all_lines_kitti_2= [   
    'model_optimization_flavor(optimization_level=3, compression_level=0, batch_size=1)',
    'model_optimization_config(checker_cfg, batch_size=1)',
    f'model_optimization_config(calibration, calibset_size={cs_size}, batch_size=1)',
    'pre_quantization_optimization(activation_clipping, layers={*}, mode=percentile, clipping_values=[0.01, 99.99])',
    f'post_quantization_optimization(finetune, policy=enabled, learning_rate=0.0001, epochs=8, dataset_size=4000)',
    'post_quantization_optimization(bias_correction, policy=enabled, batch_size=1)'
]

# all_lines_kitti_2_mod= [   
#     'model_optimization_flavor(optimization_level=3, compression_level=0, batch_size=1)',
#     'model_optimization_config(checker_cfg, batch_size=1)',
#     f'model_optimization_config(calibration, calibset_size={cs_size}, batch_size=1)',
#     'pre_quantization_optimization(activation_clipping, layers={*}, mode=percentile, clipping_values=[0.01, 99.99])',
#     f'post_quantization_optimization(finetune, policy=enabled, learning_rate=0.0001, epochs=8, dataset_size={cs_size}, batch_size=1)',
#     'post_quantization_optimization(bias_correction, policy=enabled, batch_size=1)'
# ]

all_lines_kitti_2_mod= [   #de los mejores  
    'model_optimization_flavor(optimization_level=3, compression_level=0, batch_size=1)',
    'model_optimization_config(checker_cfg, batch_size=1)',
    f'model_optimization_config(calibration, calibset_size={cs_size}, batch_size=1)',
    'pre_quantization_optimization(activation_clipping, layers={*}, mode=percentile, clipping_values=[0.01, 99.99])',
    f'post_quantization_optimization(finetune, policy=enabled, learning_rate=0.0001, epochs=8, dataset_size={cs_size}, batch_size=1)',
    'post_quantization_optimization(bias_correction, policy=enabled, batch_size=1)',
    'post_quantization_optimization(adaround, policy=enabled, batch_size=1, shuffle=False)'
]

all_lines_kitti_2_mod2= [   #algo peor
    'model_optimization_flavor(optimization_level=3, compression_level=0, batch_size=1)',
    'model_optimization_config(checker_cfg, batch_size=1)',
    f'model_optimization_config(calibration, calibset_size={cs_size}, batch_size=1)',
    #'pre_quantization_optimization(activation_clipping, layers={*}, mode=percentile, clipping_values=[0.01, 99.99])',
    f'post_quantization_optimization(finetune, policy=enabled, learning_rate=0.0001, epochs=8, dataset_size={cs_size}, batch_size=1)',
    'post_quantization_optimization(bias_correction, policy=enabled, batch_size=1)',
    'post_quantization_optimization(adaround, policy=enabled, batch_size=1, shuffle=False)'
]

all_lines_kitti_2_clipp_manual= [   #mucho peor sobre todo en pedestrian
    'model_optimization_flavor(optimization_level=3, compression_level=0, batch_size=1)',
    'model_optimization_config(checker_cfg, batch_size=1)',
    f'model_optimization_config(calibration, calibset_size={cs_size}, batch_size=1)',
    'pre_quantization_optimization(activation_clipping, layers=[conv1,conv2,conv3,conv4,conv5], mode=percentile, clipping_values=[0.01, 99.9])',
    'pre_quantization_optimization(activation_clipping, layers=[conv6,conv8], mode=percentile, clipping_values=[0.01, 99.99])',
    'pre_quantization_optimization(activation_clipping, layers=[conv7,conv9], mode=percentile, clipping_values=[0.01, 99.999])',
    f'post_quantization_optimization(finetune, policy=enabled, learning_rate=0.0001, epochs=8, dataset_size={cs_size}, batch_size=1)',
    'post_quantization_optimization(bias_correction, policy=enabled, batch_size=1)',
    'post_quantization_optimization(adaround, policy=enabled, batch_size=1, shuffle=False)'
]

all_lines_kitti_2_clipp_manual2= [   #mejora algo en car, empeora un poco en cyc mucho en pedestrian
    'model_optimization_flavor(optimization_level=3, compression_level=0, batch_size=1)',
    'model_optimization_config(checker_cfg, batch_size=1)',
    f'model_optimization_config(calibration, calibset_size={cs_size}, batch_size=1)',
    #'pre_quantization_optimization(activation_clipping, layers=[conv1,conv2,conv3,conv4,conv5], mode=percentile, clipping_values=[0.01, 99.9])',
    'pre_quantization_optimization(activation_clipping, layers=[conv1,conv2,conv3,conv4,conv5,conv6,conv8], mode=percentile, clipping_values=[0.01, 99.99])',
    'pre_quantization_optimization(activation_clipping, layers=[conv7,conv9], mode=percentile, clipping_values=[0.01, 99.999])',
    f'post_quantization_optimization(finetune, policy=enabled, learning_rate=0.0001, epochs=8, dataset_size={cs_size}, batch_size=1)',
    'post_quantization_optimization(bias_correction, policy=enabled, batch_size=1)',
    'post_quantization_optimization(adaround, policy=enabled, batch_size=1, shuffle=False)'
]


all_lines_kitti_2_clipp_manual3= [   #peor en todas las clases
    'model_optimization_flavor(optimization_level=3, compression_level=0, batch_size=1)',
    'model_optimization_config(checker_cfg, batch_size=1)',
    f'model_optimization_config(calibration, calibset_size={cs_size}, batch_size=1)',
    'pre_quantization_optimization(activation_clipping, layers=[conv1,conv2,conv3,conv4,conv5], mode=percentile, clipping_values=[0.01, 99.999])',
    'pre_quantization_optimization(activation_clipping, layers=[conv6,conv7,conv8,conv9,conv10,conv11,conv12,deconv1,conv17,deconv2], mode=percentile, clipping_values=[0.01, 99.995])',
    'pre_quantization_optimization(activation_clipping, layers=[conv13,conv14,conv15,conv16], mode=percentile, clipping_values=[0.01, 99.95])',
    'pre_quantization_optimization(activation_clipping, layers=[conv18, conv18, conv20], mode=percentile, clipping_values=[0.01, 99.99])',
    f'post_quantization_optimization(finetune, policy=enabled, learning_rate=0.0001, epochs=8, dataset_size={cs_size}, batch_size=1)',
    'post_quantization_optimization(bias_correction, policy=enabled, batch_size=1)',
    'post_quantization_optimization(adaround, policy=enabled, batch_size=1, shuffle=False)'
]


all_lines_kitti_2_clipp_manual3_finetune2= [   #sorprendentemente bastante bien igual mejor en car y ped respecto al 2 y algo peor o igual en cyc parece que el finetune extra ayuda
    'model_optimization_flavor(optimization_level=3, compression_level=0, batch_size=1)',
    'model_optimization_config(checker_cfg, batch_size=1)',
    f'model_optimization_config(calibration, calibset_size={cs_size}, batch_size=1)',
    'pre_quantization_optimization(activation_clipping, layers=[conv1,conv2,conv3,conv4,conv5], mode=percentile, clipping_values=[0.01, 99.999])',
    'pre_quantization_optimization(activation_clipping, layers=[conv6,conv7,conv8,conv9,conv10,conv11,conv12,deconv1,conv17,deconv2], mode=percentile, clipping_values=[0.01, 99.995])',
    'pre_quantization_optimization(activation_clipping, layers=[conv13,conv14,conv15,conv16], mode=percentile, clipping_values=[0.01, 99.95])',
    'pre_quantization_optimization(activation_clipping, layers=[conv18, conv18, conv20], mode=percentile, clipping_values=[0.01, 99.99])',
    f'post_quantization_optimization(finetune, policy=enabled, learning_rate=0.0001, epochs=15, dataset_size={cs_size}, batch_size=1)',
    'post_quantization_optimization(bias_correction, policy=enabled, batch_size=1)',
    'post_quantization_optimization(adaround, policy=enabled, batch_size=1, shuffle=False)'
]

all_lines_kitti_2_finetune_ext= [  #peor q sin aumentar el finetune es decir peor que 64_2_mod puede ser por las 150 epoch de adaround?? probar sin eso o quizas el fine tune beneficia mas al de clipping manual  nidea
    'model_optimization_flavor(optimization_level=3, compression_level=0, batch_size=1)',
    'model_optimization_config(checker_cfg, batch_size=1)',
    f'model_optimization_config(calibration, calibset_size={cs_size}, batch_size=1)',
    'pre_quantization_optimization(activation_clipping, layers={*}, mode=percentile, clipping_values=[0.01, 99.99])',
    f'post_quantization_optimization(finetune, policy=enabled, learning_rate=0.0001, epochs=22, dataset_size={cs_size}, batch_size=1)',
    'post_quantization_optimization(bias_correction, policy=enabled, batch_size=1)',
    'post_quantization_optimization(adaround, policy=enabled, batch_size=1, shuffle=False, cache_compression=enabled, epochs=150)'
]

all_lines_kitti_custom= [    #na no mejora
    'model_optimization_flavor(optimization_level=3, compression_level=0, batch_size=1)',
    'model_optimization_config(checker_cfg, batch_size=1)',
    f'model_optimization_config(calibration, calibset_size={cs_size}, batch_size=1)',
    'pre_quantization_optimization(activation_clipping, layers={*}, mode=percentile, clipping_values=[0.01, 99.99])',
    f'post_quantization_optimization(finetune, policy=enabled, learning_rate=0.0001, epochs=15, dataset_size={finetune_size}, batch_size=1)',
    f'post_quantization_optimization(bias_correction, policy=enabled, batch_size=1)',
    f'post_quantization_optimization(adaround, policy=enabled, dataset_size={adaraound_size}, batch_size=1, cache_compression=enabled)'
]

all_lines_kitti_2_clipp_manual3_finetune3= [   #respecto al top se mantiene en car , en cyc mejora y en ped empeora, 64 y 20 epochs
    'model_optimization_flavor(optimization_level=3, compression_level=0, batch_size=1)',
    'model_optimization_config(checker_cfg, batch_size=1)',
    f'model_optimization_config(calibration, calibset_size={cs_size}, batch_size=1)',
    'pre_quantization_optimization(activation_clipping, layers=[conv1,conv2,conv3,conv4,conv5], mode=percentile, clipping_values=[0.01, 99.999])',
    'pre_quantization_optimization(activation_clipping, layers=[conv6,conv7,conv8,conv9,conv10,conv11,conv12,deconv1,conv17,deconv2], mode=percentile, clipping_values=[0.01, 99.995])',
    'pre_quantization_optimization(activation_clipping, layers=[conv13,conv14,conv15,conv16], mode=percentile, clipping_values=[0.01, 99.95])',
    'pre_quantization_optimization(activation_clipping, layers=[conv18, conv18, conv20], mode=percentile, clipping_values=[0.01, 99.99])',
    f'post_quantization_optimization(finetune, policy=enabled, learning_rate=0.0001, epochs=15, dataset_size={finetune_size}, batch_size=1)',
    'post_quantization_optimization(bias_correction, policy=enabled, batch_size=1)',
    f'post_quantization_optimization(adaround, dataset_size={adaraound_size}, policy=enabled, batch_size=1, shuffle=False)'
]

all_lines_kitti_2_clipp_manual3_finetune4= [    #64 sale bien incluso algo mejor, pero al revisar el profiler lo q refleja es menos consistente
    'model_optimization_flavor(optimization_level=3, compression_level=0, batch_size=1)',
    'model_optimization_config(checker_cfg, batch_size=1)',
    f'model_optimization_config(calibration, calibset_size={cs_size}, batch_size=1)',
    
    'pre_quantization_optimization(activation_clipping, layers=[conv1,conv2,conv3,conv4,conv5], mode=percentile, clipping_values=[0.01, 99.999])',
    'pre_quantization_optimization(activation_clipping, layers=[conv6,conv7,conv8,conv9,conv10,conv11,conv12,deconv1,conv17,deconv2], mode=percentile, clipping_values=[0.01, 99.995])',
    'pre_quantization_optimization(activation_clipping, layers=[conv13,conv14,conv15,conv16], mode=percentile, clipping_values=[0.01, 99.95])',
    'pre_quantization_optimization(activation_clipping, layers=[conv18, conv18, conv20], mode=percentile, clipping_values=[0.01, 99.99])',
    
    f'post_quantization_optimization(finetune, policy=enabled, learning_rate=0.0001, epochs=20, dataset_size={finetune_size}, batch_size=1)',
    'post_quantization_optimization(bias_correction, policy=enabled, batch_size=1)',
    f'post_quantization_optimization(adaround, dataset_size={adaraound_size}, policy=enabled, batch_size=1, shuffle=False)'
]

all_lines_kitti_final2= [   
    'model_optimization_flavor(optimization_level=3, compression_level=0, batch_size=1)',
    'model_optimization_config(checker_cfg, batch_size=1)',
    f'model_optimization_config(calibration, calibset_size={cs_size}, batch_size=1)',
    
    # 'pre_quantization_optimization(activation_clipping, layers=[conv1,conv2,conv3,conv4,conv5], mode=percentile, clipping_values=[0.01, 99.999])',
    # 'pre_quantization_optimization(activation_clipping, layers=[conv6,conv7,conv8,conv9,conv10,conv11,conv12,deconv1,conv17,deconv2], mode=percentile, clipping_values=[0.01, 99.995])',
    # 'pre_quantization_optimization(activation_clipping, layers=[conv13,conv14,conv15,conv16], mode=percentile, clipping_values=[0.01, 99.95])',
    # 'pre_quantization_optimization(activation_clipping, layers=[conv18, conv18, conv20], mode=percentile, clipping_values=[0.01, 99.99])',
    
    f'post_quantization_optimization(mix_precision_search, policy=enabled, batch_size=1, dataset_size={adaraound_size})',
    f'post_quantization_optimization(finetune, policy=enabled, learning_rate=0.0001, epochs=20, dataset_size={finetune_size}, batch_size=1)',
    'post_quantization_optimization(bias_correction, policy=enabled, batch_size=1)',
    f'post_quantization_optimization(adaround, dataset_size={adaraound_size}, policy=enabled, batch_size=1, shuffle=False)'
    

]

all_lines_kitti_final3 = [
    'model_optimization_flavor(optimization_level=3, compression_level=3, batch_size=1)',
    # Fuerza el uso de 4 bits en ~60 % de los pesos:
    'model_optimization_config(compression_params, auto_4bit_weights_ratio=0.6)',
    'model_optimization_config(checker_cfg, batch_size=1)',
    f'model_optimization_config(calibration, calibset_size={cs_size}, batch_size=1)',

    f'post_quantization_optimization(mix_precision_search, policy=enabled, dataset_size={adaraound_size}, batch_size=1, snr_cap=20, optimizer=linear, comprecision_metric=bops)',
    f'post_quantization_optimization(finetune, policy=enabled, learning_rate=0.0001, epochs=20, dataset_size={finetune_size}, batch_size=1)',
    'post_quantization_optimization(bias_correction, policy=enabled, batch_size=1)',
    f'post_quantization_optimization(adaround, dataset_size={adaraound_size}, policy=enabled, batch_size=1, shuffle=False)'
]

all_lines_kitti_final4 = [
    'model_optimization_flavor(optimization_level=3, compression_level=4, batch_size=1)',
    # Fuerza el uso de 4 bits en ~60 % de los pesos:
    'model_optimization_config(compression_params, auto_4bit_weights_ratio=1)',
    'model_optimization_config(checker_cfg, batch_size=1)',
    f'model_optimization_config(calibration, calibset_size={cs_size}, batch_size=1)',

    'pre_quantization_optimization(activation_clipping, layers=[conv1,conv2,conv3,conv4,conv5], mode=percentile, clipping_values=[0.01, 99.999])',
    'pre_quantization_optimization(activation_clipping, layers=[conv6,conv7,conv8,conv9,conv10,conv11,conv12,deconv1,conv17,deconv2], mode=percentile, clipping_values=[0.01, 99.995])',
    'pre_quantization_optimization(activation_clipping, layers=[conv13,conv14,conv15,conv16], mode=percentile, clipping_values=[0.01, 99.95])',
    'pre_quantization_optimization(activation_clipping, layers=[conv18, conv18, conv20], mode=percentile, clipping_values=[0.01, 99.99])',
    
    #f'post_quantization_optimization(mix_precision_search, policy=enabled, dataset_size={adaraound_size}, batch_size=1, snr_cap=20, optimizer=linear, comprecision_metric=bops)',
    f'post_quantization_optimization(finetune, policy=enabled, learning_rate=0.0001, epochs=20, dataset_size={finetune_size}, batch_size=1)',
    'post_quantization_optimization(bias_correction, policy=enabled, batch_size=1)',
    f'post_quantization_optimization(adaround, dataset_size={adaraound_size}, policy=enabled, batch_size=1, shuffle=False)'
]



all_lines_kitti_final_fast= [    #64 resultados: igual, bien en regla general
    'model_optimization_flavor(optimization_level=3, compression_level=0, batch_size=1)',
    'model_optimization_config(checker_cfg, batch_size=1)',
    f'model_optimization_config(calibration, calibset_size={cs_size}, batch_size=1)',
    

    # --- Pre-cuánticas ---
    'pre_quantization_optimization(equalization, policy=enabled)',
    'pre_quantization_optimization(dead_layers_removal, policy=enabled)',
    'pre_quantization_optimization(weights_clipping, layers={conv*}, mode=mmse)',

    'pre_quantization_optimization(activation_clipping, layers=[conv1,conv2,conv3,conv4,conv5], mode=percentile, clipping_values=[0.01, 99.999])',
    'pre_quantization_optimization(activation_clipping, layers=[conv6,conv7,conv8,conv9,conv10,conv11,conv12,deconv1,conv17,deconv2], mode=percentile, clipping_values=[0.01, 99.995])',
    'pre_quantization_optimization(activation_clipping, layers=[conv13,conv14,conv15,conv16], mode=percentile, clipping_values=[0.01, 99.95])',
    'pre_quantization_optimization(activation_clipping, layers=[conv18, conv18, conv20], mode=percentile, clipping_values=[0.01, 99.99])',
    
    f'post_quantization_optimization(finetune, policy=enabled, learning_rate=0.0001, epochs=20, dataset_size={finetune_size}, batch_size=1)',
    'post_quantization_optimization(bias_correction, policy=enabled, batch_size=1)',
    f'post_quantization_optimization(adaround, dataset_size={adaraound_size}, policy=enabled, batch_size=1, shuffle=False)'
]

all_lines_kitti_final_fast2= [    #doing:fatal q me meo, puede ser por el weights clipping?
    'model_optimization_flavor(optimization_level=3, compression_level=0, batch_size=1)',
    'model_optimization_config(checker_cfg, batch_size=1)',
    f'model_optimization_config(calibration, calibset_size={cs_size}, batch_size=1)',
    
    # --- Pre-cuánticas ---
    'pre_quantization_optimization(equalization, policy=enabled)',
    'pre_quantization_optimization(dead_layers_removal, policy=enabled)',
    'pre_quantization_optimization(weights_clipping, layers={*}, mode=mmse)',

    'pre_quantization_optimization(activation_clipping, layers=[conv1,conv2,conv3,conv4,conv5,conv6,conv7,conv8,conv9,conv10,conv11,conv12], mode=percentile, clipping_values=[0.01, 99.999])',
    'pre_quantization_optimization(activation_clipping, layers=[deconv1,conv17,deconv2], mode=percentile, clipping_values=[0.01, 99.995])',
    'pre_quantization_optimization(activation_clipping, layers=[conv13,conv14,conv15,conv16], mode=percentile, clipping_values=[0.01, 99.99])',
    'pre_quantization_optimization(activation_clipping, layers=[conv18, conv19, conv20], mode=percentile, clipping_values=[0.01, 99.99])',
    
    f'post_quantization_optimization(finetune, policy=enabled, learning_rate=0.0001, epochs=20, dataset_size={finetune_size}, batch_size=1)',
    'post_quantization_optimization(bias_correction, policy=enabled, batch_size=1)',
    f'post_quantization_optimization(adaround, dataset_size={adaraound_size}, policy=enabled, batch_size=1, shuffle=False)'
]

all_lines_kitti_final_fast3= [   #resultados fatal
    'model_optimization_flavor(optimization_level=3, compression_level=0, batch_size=1)',
    'model_optimization_config(checker_cfg, batch_size=1)',
    f'model_optimization_config(calibration, calibset_size={cs_size}, batch_size=1)',
    
    # --- Pre-cuánticas ---
    'pre_quantization_optimization(equalization, policy=enabled)',
    'pre_quantization_optimization(dead_layers_removal, policy=enabled)',
    'pre_quantization_optimization(weights_clipping, layers={*}, mode=mmse)',

    'pre_quantization_optimization(activation_clipping, layers=[conv1,conv2,conv3,conv4,conv5,conv7,conv8,conv9,conv10,conv11], mode=percentile, clipping_values=[0.01, 99.999])',
    'pre_quantization_optimization(activation_clipping, layers=[deconv1,conv17,deconv2,conv13,conv14,conv15,conv16], mode=percentile, clipping_values=[0.01, 99.995])',
    'pre_quantization_optimization(activation_clipping, layers=[conv18, conv19, conv20], mode=percentile, clipping_values=[0.01, 99.99])',
    
    f'post_quantization_optimization(finetune, policy=enabled, learning_rate=0.0001, epochs=20, dataset_size={finetune_size}, batch_size=1)',
    'post_quantization_optimization(bias_correction, policy=enabled, batch_size=1)',
    f'post_quantization_optimization(adaround, dataset_size={adaraound_size}, policy=enabled, batch_size=1, shuffle=False)'
]