from util import init
from dataset import data
from util.config import config

# As user, you can either set separate languages or a language family
languages = ["nld","deu"]
lang_family = None

lang_family_dict = {
"slav": ["ces", "bul", "rus", "bel", "ukr", "pol", "slk", "slv", "hrv"],
"ger": ["swe", "isl", "eng", "nld", "deu", "dan", "nor"]
}
if lang_family:
    languages = lang_family_dict[lang_family]

options, distances_path, baselines_path = init.initialize_program()
results_path, lang_pairs, train, val, test, max_len, conversion_key, voc_size = data.load_data(train_corpus="northeuralex",
                                                                                               valtest_corpus="northeuralex",
                                                                                               languages=languages,  
                                                                                               input_type="asjp", 
                                                                                               options=options)
