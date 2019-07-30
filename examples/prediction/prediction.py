from models.EncoderDecoder import EncoderDecoder
from models.PhylNet import PhylNet
from models.SeqModel import SeqModel

def word_prediction(lang_a, lang_b, max_len, train, val, test, conversion_key, voc_size, results_path, distances_path, context_vectors_path, FLAGS):
    
    print("Create RNN instance.")
    net = EncoderDecoder(batch_size=FLAGS.batch_size,
                        max_len=max_len,
                        voc_size=voc_size,
                        n_hidden=FLAGS.hidden,
                        n_layers_encoder=FLAGS.layers_encoder,
                        n_layers_decoder=FLAGS.layers_decoder,
                        n_layers_dense=FLAGS.layers_dense,
                        bidirectional_encoder=not FLAGS.no_bidirectional_encoder,
                        bidirectional_decoder=FLAGS.bidirectional_decoder,
                        encoder_only_final=not FLAGS.encoder_all_steps,
                        dropout=FLAGS.dropout,
                        learning_rate=FLAGS.learning_rate,
                        learning_rate_decay=FLAGS.lr_decay,
                        adaptive_learning_rate=FLAGS.adaptive_lr,
                        reg_weight=FLAGS.reg_weight,
                        grad_clip=FLAGS.grad_clip,
                        initialization=FLAGS.init,
                        gated_layer_type=FLAGS.gated_layer_type,
                        cognacy_prior=FLAGS.cognacy_prior,
                        input_encoding=FLAGS.input_encoding,
                        output_encoding=OUTPUT_ENCODING,
                        conversion_key=conversion_key,
                        export_weights=FLAGS.export_weights,
                        optimizer=FLAGS.optimizer)
    
    # Both prediction and validation rounds during training are performed on 'testset'
    # which can be the test set or the validation set
    if FLAGS.validation:
        testset = val
    else:
        testset = test
    
    # Train network on train set
    losses, distances = net.train(trainset=train,
              valset=testset,
              n_epochs=FLAGS.n_epochs)
    
    # Predict on testset (which can be validation set if FLAGS.validation)
    print("Predict and show results.")
    avg_distance, results_table, context_vectors = net.predict(testset=testset, max_len_tar=max_len[1], voc_size_tar=voc_size[1], conversion_key=conversion_key, predict_func=net.predict_func)
    data.write_results_table(results_table, testset=testset, results_filename=results_path + ".tsv")
    
    # Plot loss
    utility.plot_loss(losses, distances, results_path + ".png")
    
    # Write distances to file
    with open(distances_path, "a") as f:
        f.write(lang_a + "," + lang_b + "," + str(avg_distance) + "\n")
    
    # Write context vectors to pickle file
    if FLAGS.export_weights:
        with open(context_vectors_path, "wb") as f:
            pickle.dump(context_vectors, f)


def word_prediction_seq(lang_a, lang_b, train, val, test, conversion_key, results_path, distances_path, FLAGS):
    
    print("Create SeqModel instance.")
    model = SeqModel(input_encoding=FLAGS.input_encoding,
              conversion_key=conversion_key, n_iter_seq=FLAGS.n_iter_seq)
    
    # Both prediction and validation rounds during training are performed on 'testset'
    # which can be the test set or the validation set
    if FLAGS.validation:
        testset = val
    else:
        testset = test
    
    # Train network on train set
    print("Train sequential model.")
    model.train(train)
    
    # Predict on testset (which can be validation set if FLAGS.validation)
    print("Predict using sequential model and show results.")
    avg_distance, results_table = model.predict(testset=testset)
    data.write_results_table(results_table, testset=testset, results_filename=results_path + ".tsv")
    
    # Write distances to file
    with open(distances_path, "a") as f:
        f.write(lang_a + "," + lang_b + "," + str(avg_distance) + "\n")


def word_prediction_phyl(languages, lang_pairs, tree_string, max_len, train, val, test, conversion_key, voc_size, results_path, results_path_proto, distances_path, context_vectors_path, plot_path, FLAGS):
    
    print("Create phylogenetic network instance.")
    net = PhylNet(langs=languages, lang_pairs=lang_pairs, batch_size=FLAGS.batch_size,
                        tree_string=tree_string,
                        max_len=max_len,
                        voc_size=voc_size,
                        n_hidden=FLAGS.hidden,
                        n_layers_encoder=FLAGS.layers_encoder,
                        n_layers_decoder=FLAGS.layers_decoder,
                        n_layers_dense=FLAGS.layers_dense,
                        bidirectional_encoder=not FLAGS.no_bidirectional_encoder,
                        bidirectional_decoder=FLAGS.bidirectional_decoder,
                        encoder_only_final=not FLAGS.encoder_all_steps,
                        dropout=FLAGS.dropout,
                        learning_rate=FLAGS.learning_rate,
                        learning_rate_decay=FLAGS.lr_decay,
                        adaptive_learning_rate=FLAGS.adaptive_lr,
                        reg_weight=FLAGS.reg_weight,
                        grad_clip=FLAGS.grad_clip,
                        initialization=FLAGS.init,
                        gated_layer_type=FLAGS.gated_layer_type,
                        cognacy_prior=FLAGS.cognacy_prior,
                        input_encoding=FLAGS.input_encoding,
                        output_encoding=OUTPUT_ENCODING,
                        conversion_key=conversion_key,
                        export_weights=FLAGS.export_weights,
                        optimizer=FLAGS.optimizer,
                        units_phyl=FLAGS.units_phyl,
                        train_proto=FLAGS.train_proto)
    
    proto_languages = net.get_proto_languages()
    for lang_in in FLAGS.languages:
        for lang_proto in proto_languages:
            # We have to use one of the existing lang pairs, of which we discard the second language
            # This is a bit of a hack.
            used_lang_pair = utility.find(lambda x: x[0] == lang_in, lang_pairs)
            train[(lang_in, lang_proto)] = copy.deepcopy(train[used_lang_pair])
            val[(lang_in, lang_proto)] = copy.deepcopy(val[used_lang_pair])
            test[(lang_in, lang_proto)] = copy.deepcopy(test[used_lang_pair])
    
    # Both prediction and validation rounds during training are performed on 'testset'
    # which can be the test set or the validation set
    if FLAGS.validation:
        testset = val
    else:
        testset = test
    
    # Train network on train set
    losses, distances = net.train_all(trainset=train,
              valset=testset,
              n_epochs=FLAGS.n_epochs)
    # Plot loss
    # print("Losses:")
    # print(losses)
    # print("Distances:")
    # print(distances)
    utility.plot_loss_phyl(losses, distances, plot_path)
    
    # Predict on testset (which can be validation set if FLAGS.validation)
    print("Predicting protoforms...")
    for lang_in in FLAGS.languages:
        for lang_proto in proto_languages:
            print("Predicting protoform for " + lang_proto + " from " + lang_in)
            # Use max_len of input language
            avg_distance, results_table, context_vectors = net.predict_proto(testset=testset[(lang_in, lang_proto)], max_len_tar=max_len[lang_in], voc_size_tar=voc_size[1], conversion_key=conversion_key, predict_func=net.get_predict_func(lang_in, lang_proto))
            data.write_results_table(results_table, testset=testset, results_filename=results_path_proto + "-" + lang_in + "-" + lang_proto + ".tsv")
    
    # Predict on testset (which can be validation set if FLAGS.validation)
    print("Predict and show results.")
    for lang_a, lang_b in sorted(lang_pairs):
        avg_distance, results_table, context_vectors = net.predict(testset=testset[(lang_a, lang_b)], max_len_tar=max_len[lang_b], voc_size_tar=voc_size[1], conversion_key=conversion_key, predict_func=net.get_predict_func(lang_a, lang_b), print_output=False)
        data.write_results_table(results_table, testset=testset, results_filename=results_path[(lang_a, lang_b)] + ".tsv")
        
        # Write distances to file
        with open(distances_path, "a") as f:
            f.write(lang_a + "," + lang_b + "," + str(avg_distance) + "\n")
        
        # Write context vectors to pickle file
        if FLAGS.export_weights:
            with open(context_vectors_path[(lang_a, lang_b)], "wb") as f:
                pickle.dump(context_vectors, f)
    with open(distances_path, "r") as f:
        print(f.read())