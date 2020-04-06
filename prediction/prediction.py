from models.EncoderDecoder import EncoderDecoder
from models.PhylNet import PhylNet
from models.SeqModel import SeqModel
from dataset import data
from util import utility

import copy
import pickle


def word_prediction(lang_a, lang_b, max_len, train, val, test, conversion_key, voc_size, results_path,
                    distances_path, context_vectors_path, output_encoding, config):
    print("Create RNN instance.")
    net = EncoderDecoder(batch_size=config["batch_size"],
                         max_len=max_len,
                         voc_size=voc_size,
                         n_hidden=config["n_hidden"],
                         n_layers_encoder=config["n_layers_encoder"],
                         n_layers_decoder=config["n_layers_decoder"],
                         n_layers_dense=config["n_layers_dense"],
                         bidirectional_encoder=config["bidirectional_encoder"],
                         bidirectional_decoder=config["bidirectional_decoder"],
                         encoder_only_final=not config["encoder_all_steps"],
                         dropout=config["dropout"],
                         learning_rate=config["learning_rate"],
                         learning_rate_decay=config["lr_decay"],
                         adaptive_learning_rate=config["adaptive_lr"],
                         reg_weight=config["reg_weight"],
                         grad_clip=config["grad_clip"],
                         initialization=config["init"],
                         gated_layer_type=config["gated_layer_type"],
                         cognacy_prior=config["cognacy_prior"],
                         input_encoding=config["input_encoding"],
                         output_encoding=output_encoding,
                         conversion_key=conversion_key,
                         export_weights=config["export_weights"],
                         optimizer=config["optimizer"])

    # Both prediction and validation rounds during training are performed on 'testset'
    # which can be the test set or the validation set
    if config["validation"]:
        testset = val
    else:
        testset = test

    # Train network on train set
    losses, distances = net.train(trainset=train,
                                  valset=testset,
                                  n_epochs=config["n_epochs"])

    # Predict on testset (which can be validation set if config["validation)
    print("Predict and show results.")
    avg_distance, results_table, context_vectors = net.predict(
        testset=testset, max_len_tar=max_len[1], voc_size_tar=voc_size[1],
        conversion_key=conversion_key, predict_func=net.predict_func)
    data.write_results_table(results_table, testset=testset, results_filename=results_path + ".tsv")

    # Plot loss
    utility.plot_loss(losses, distances, results_path + ".png")

    # Write distances to file
    with open(distances_path, "a") as f:
        f.write(lang_a + "," + lang_b + "," + str(avg_distance) + "\n")

    # Write context vectors to pickle file
    if config["export_weights"]:
        with open(context_vectors_path, "wb") as f:
            pickle.dump(context_vectors, f)


def word_prediction_seq(lang_a, lang_b, train, val, test, conversion_key, results_path, distances_path, config):

    print("Create SeqModel instance.")
    model = SeqModel(input_encoding=config["input_encoding"],
                     conversion_key=conversion_key, n_iter_seq=config["n_iter_seq"])

    # Both prediction and validation rounds during training are performed on 'testset'
    # which can be the test set or the validation set
    if config["validation"]:
        testset = val
    else:
        testset = test

    # Train network on train set
    print("Train sequential model.")
    model.train(train)

    # Predict on testset (which can be validation set if config["validation)
    print("Predict using sequential model and show results.")
    avg_distance, results_table = model.predict(testset=testset)
    data.write_results_table(results_table, testset=testset, results_filename=results_path + ".tsv")

    # Write distances to file
    with open(distances_path, "a") as f:
        f.write(lang_a + "," + lang_b + "," + str(avg_distance) + "\n")


def word_prediction_phyl(languages, lang_pairs, tree_string, max_len, train, val, test, conversion_key, voc_size,
                         results_path, results_path_proto, distances_path, context_vectors_path, plot_path,
                         output_encoding, config):

    print("Creat  e phylogenetic network instance.")
    net = PhylNet(langs=languages, lang_pairs=lang_pairs, batch_size=config["batch_size"],
                  tree_string=tree_string,
                  max_len=max_len,
                  voc_size=voc_size,
                  n_hidden=config["n_hidden"],
                  n_layers_encoder=config["n_layers_encoder"],
                  n_layers_decoder=config["n_layers_decoder"],
                  n_layers_dense=config["n_layers_dense"],
                  bidirectional_encoder=not config["no_bidirectional_encoder"],
                  bidirectional_decoder=config["bidirectional_decoder"],
                  encoder_only_final=not config["encoder_all_steps"],
                  dropout=config["dropout"],
                  learning_rate=config["learning_rate"],
                  learning_rate_decay=config["lr_decay"],
                  adaptive_learning_rate=config["adaptive_lr"],
                  reg_weight=config["reg_weight"],
                  grad_clip=config["grad_clip"],
                  initialization=config["init"],
                  gated_layer_type=config["gated_layer_type"],
                  cognacy_prior=config["cognacy_prior"],
                  input_encoding=config["input_encoding"],
                  output_encoding=output_encoding,
                  conversion_key=conversion_key,
                  export_weights=config["export_weights"],
                  optimizer=config["optimizer"],
                  units_phyl=config["units_phyl"],
                  train_proto=config["train_proto"])

    proto_languages = net.get_proto_languages()
    for lang_in in config["languages"]:
        for lang_proto in proto_languages:
            # We have to use one of the existing lang pairs, of which we discard the second language
            # This is a bit of a hack.
            used_lang_pair = utility.find(lambda x: x[0] == lang_in, lang_pairs)
            train[(lang_in, lang_proto)] = copy.deepcopy(train[used_lang_pair])
            val[(lang_in, lang_proto)] = copy.deepcopy(val[used_lang_pair])
            test[(lang_in, lang_proto)] = copy.deepcopy(test[used_lang_pair])

    # Both prediction and validation rounds during training are performed on 'testset'
    # which can be the test set or the validation set
    if config["validation"]:
        testset = val
    else:
        testset = test

    # Train network on train set
    losses, distances = net.train_all(trainset=train,
                                      valset=testset,
                                      n_epochs=config["n_epochs"])
    # Plot loss
    # print("Losses:")
    # print(losses)
    # print("Distances:")
    # print(distances)
    utility.plot_loss_phyl(losses, distances, plot_path)

    # Predict on testset (which can be validation set if config["validation)
    print("Predicting protoforms...")
    for lang_in in config["languages"]:
        for lang_proto in proto_languages:
            print("Predicting protoform for " + lang_proto + " from " + lang_in)
            # Use max_len of input language
            avg_distance, results_table, context_vectors = net.predict_proto(testset=testset[(
                lang_in, lang_proto)], max_len_tar=max_len[lang_in], voc_size_tar=voc_size[1],
                conversion_key=conversion_key, predict_func=net.get_predict_func(lang_in, lang_proto))
            data.write_results_table(results_table, testset=testset,
                                     results_filename=results_path_proto + "-" + lang_in + "-" + lang_proto + ".tsv")

    # Predict on testset (which can be validation set if config["validation)
    print("Predict and show results.")
    for lang_a, lang_b in sorted(lang_pairs):
        avg_distance, results_table, context_vectors = net.predict(testset=testset[(
            lang_a, lang_b)], max_len_tar=max_len[lang_b], voc_size_tar=voc_size[1],
            conversion_key=conversion_key, predict_func=net.get_predict_func(lang_a, lang_b), print_output=False)
        data.write_results_table(results_table, testset=testset,
                                 results_filename=results_path[(lang_a, lang_b)] + ".tsv")

        # Write distances to file
        with open(distances_path, "a") as f:
            f.write(lang_a + "," + lang_b + "," + str(avg_distance) + "\n")

        # Write context vectors to pickle file
        if config["export_weights"]:
            with open(context_vectors_path[(lang_a, lang_b)], "wb") as f:
                pickle.dump(context_vectors, f)
    with open(distances_path, "r") as f:
        print(f.read())
