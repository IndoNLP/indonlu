import os
from tabulate import tabulate
from argparse import ArgumentParser

import pandas as pd
pd.options.display.float_format = '{:,.4f}'.format

def compute_num_params(model):
    """
    Computes number of trainable and non-trainable parameters
    """
    sizes = [(np.array(p.data.size()).prod(), int(p.requires_grad)) for p in model.parameters()]
    return sum(map(lambda t: t[0]*t[1], sizes)), sum(map(lambda t: t[0]*(1 - t[1]), sizes))

def print_opts(opts):
    """Prints the values of all command-line arguments.
    """
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    for key in opts.keys():
        if opts[key]:
            print('{:>30}: {:<50}'.format(key, opts[key]).center(80))
    print('=' * 80)

def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--task", type=str, default="", help="Task name")
    parser.add_argument("--all", action='store_true', help="all")
    args = vars(parser.parse_args())
    print_opts(args)
    return args

args = get_parser()

rows = []
# rows = [
# {'task': 'emotion-twitter', 'model': 'bert-base-multilingual-uncased_lr1e-5_l12', 'acc': 0.6590909090909091, 'f1': 0.6729516946979527, 'rec': 0.6630909956380253, 'pre': 0.6929251803452052},
# {'task': 'emotion-twitter', 'model': 'babert-opensubtitle_lr1e-5_l12', 'acc': 0.6977272727272728, 'f1': 0.7089606535810445, 'rec': 0.7069314373745067, 'pre': 0.7119979182930003},
# {'task': 'emotion-twitter', 'model': 'babert-base-512_lr1e-5_l12', 'acc': 0.7613636363636364, 'f1': 0.7661079802424068, 'rec': 0.7633467510212559, 'pre': 0.770998810480996},
# {'task': 'emotion-twitter', 'model': 'xlm-roberta-large_lr1e-5_l24', 'acc': 0.7795454545454545, 'f1': 0.7851205391511296, 'rec': 0.7811157827321193, 'pre': 0.7981437291897893},
# {'task': 'emotion-twitter', 'model': 'scratch_lr6.25e-5_l4', 'acc': 0.5659090909090909, 'f1': 0.5830467142517947, 'rec': 0.5751608218514159, 'pre': 0.5953073109671579},
# {'task': 'emotion-twitter', 'model': 'scratch_lr6.25e-5_l6', 'acc': 0.5590909090909091, 'f1': 0.5734305741425396, 'rec': 0.5624692844284429, 'pre': 0.6138024691358025},
# {'task': 'emotion-twitter', 'model': 'babert-bpe-mlm-uncased-128-dup10-5_lr1e-5_l12', 'acc': 0.7568181818181818, 'f1': 0.7638929088558344, 'rec': 0.7613610399501489, 'pre': 0.7703464671333274},
# {'task': 'emotion-twitter', 'model': 'scratch_lr6.25e-5_l2', 'acc': 0.5545454545454546, 'f1': 0.5641618085927378, 'rec': 0.5621191667243648, 'pre': 0.5765886551564344},
# {'task': 'emotion-twitter', 'model': 'cartobert_lr1e-5_l12', 'acc': 0.5840909090909091, 'f1': 0.593387249478331, 'rec': 0.5888257979644119, 'pre': 0.6012622010982667},
# {'task': 'emotion-twitter', 'model': 'xlm-roberta-base_lr1e-5_l12', 'acc': 0.7181818181818181, 'f1': 0.7253491933261939, 'rec': 0.722838278058575, 'pre': 0.7373397991870736},
# {'task': 'emotion-twitter', 'model': 'xlm-mlm-100-1280_lr1e-5_l24', 'acc': 0.6477272727272727, 'f1': 0.6575293655949361, 'rec': 0.6535083258325833, 'pre': 0.6651669861363836},
# {'task': 'emotion-twitter', 'model': 'babert-bpe-mlm-large-uncased_lr1e-5_l12', 'acc': 0.775, 'f1': 0.7742600962241658, 'rec': 0.7742877172332617, 'pre': 0.7789313380294205},
# {'task': 'emotion-twitter', 'model': 'babert-bpe-mlm-large-uncased-1m_lr1e-5_l12', 'acc': 0.7818181818181819, 'f1': 0.7877220421263711, 'rec': 0.7852730215329226, 'pre': 0.793860630272131},
# {'task': 'emotion-twitter', 'model': 'albert-base-uncased-96000_lr1e-5_l12', 'acc': 0.7204545454545455, 'f1': 0.7299711051685728, 'rec': 0.720047497057398, 'pre': 0.7448661775272696},
# {'task': 'emotion-twitter', 'model': 'albert-base-uncased-112500_lr1e-5_l12', 'acc': 0.7295454545454545, 'f1': 0.7387537890689222, 'rec': 0.7348152824897876, 'pre': 0.7471563376904446},
# {'task': 'entailment-ui', 'model': 'bert-base-multilingual-uncased_lr1e-5_l12', 'acc': 0.86, 'f1': 0.8440285204991087, 'rec': 0.8297604035308953, 'pre': 0.8810101991257893},
# {'task': 'entailment-ui', 'model': 'babert-opensubtitle_lr1e-5_l12', 'acc': 0.74, 'f1': 0.7348021215830274, 'rec': 0.7452711223203027, 'pre': 0.7342432757928543},
# {'task': 'entailment-ui', 'model': 'babert-base-512_lr1e-5_l12', 'acc': 0.81, 'f1': 0.7973333333333333, 'rec': 0.7934005884825557, 'pre': 0.8029513888888888},
# {'task': 'entailment-ui', 'model': 'xlm-roberta-large_lr1e-5_l24', 'acc': 0.85, 'f1': 0.8382051558623664, 'rec': 0.8308112652374948, 'pre': 0.8507130124777184},
# {'task': 'entailment-ui', 'model': 'scratch_lr6.25e-5_l4', 'acc': 0.64, 'f1': 0.5714285714285714, 'rec': 0.5800756620428751, 'pre': 0.6148282097649186},
# {'task': 'entailment-ui', 'model': 'scratch_lr6.25e-5_l6', 'acc': 0.65, 'f1': 0.5872154735228211, 'rec': 0.592896174863388, 'pre': 0.6287878787878788},
# {'task': 'entailment-ui', 'model': 'babert-bpe-mlm-uncased-128-dup10-5_lr1e-5_l12', 'acc': 0.8, 'f1': 0.782986111111111, 'rec': 0.7759562841530054, 'pre': 0.7969244685662595},
# {'task': 'entailment-ui', 'model': 'scratch_lr6.25e-5_l2', 'acc': 0.63, 'f1': 0.6129302228266554, 'rec': 0.6134930643127364, 'pre': 0.6125},
# {'task': 'entailment-ui', 'model': 'cartobert_lr1e-5_l12', 'acc': 0.74, 'f1': 0.7241086587436332, 'rec': 0.7221521647751157, 'pre': 0.7267267267267268},
# {'task': 'entailment-ui', 'model': 'xlm-roberta-base_lr1e-5_l12', 'acc': 0.82, 'f1': 0.7994652406417111, 'rec': 0.7877259352669189, 'pre': 0.832442933462846},
# {'task': 'entailment-ui', 'model': 'xlm-mlm-100-1280_lr1e-5_l24', 'acc': 0.68, 'f1': 0.64349376114082, 'rec': 0.6406052963430013, 'pre': 0.6624575036425449},
# {'task': 'entailment-ui', 'model': 'babert-bpe-mlm-large-uncased_lr1e-5_l12', 'acc': 0.83, 'f1': 0.8062678062678061, 'rec': 0.7912988650693569, 'pre': 0.8601871101871101},
# {'task': 'entailment-ui', 'model': 'babert-bpe-mlm-large-uncased-1m_lr1e-5_l12', 'acc': 0.84, 'f1': 0.8190863862505653, 'rec': 0.8041193778898696, 'pre': 0.8670725520040589},
# {'task': 'entailment-ui', 'model': 'albert-base-uncased-96000_lr1e-5_l12', 'acc': 0.85, 'f1': 0.8316687240489282, 'rec': 0.8169398907103824, 'pre': 0.8740079365079365},
# {'task': 'entailment-ui', 'model': 'albert-base-uncased-112500_lr1e-5_l12', 'acc': 0.84, 'f1': 0.82174688057041, 'rec': 0.8087431693989071, 'pre': 0.8567265662943176},
# {'task': 'ner-grit', 'model': 'bert-base-multilingual-uncased_lr1e-5_l12', 'acc': 0.9518618150927368, 'f1': 0.7755960729312763, 'rec': 0.8037790697674418, 'pre': 0.7894361170592434},
# {'task': 'ner-grit', 'model': 'babert-opensubtitle_lr1e-5_l12', 'acc': 0.9116522724054934, 'f1': 0.5752551020408163, 'rec': 0.6555232558139535, 'pre': 0.6127717391304348},
# {'task': 'ner-grit', 'model': 'babert-base-512_lr1e-5_l12', 'acc': 0.9395441030723488, 'f1': 0.7374810318664643, 'rec': 0.7063953488372093, 'pre': 0.7216035634743875},
# {'task': 'ner-grit', 'model': 'xlm-roberta-large_lr1e-5_l24', 'acc': 0.9576667138609656, 'f1': 0.8157142857142857, 'rec': 0.8299418604651163, 'pre': 0.8227665706051872},
# {'task': 'ner-grit', 'model': 'scratch_lr6.25e-5_l4', 'acc': 0.819481806597763, 'f1': 0.3333333333333333, 'rec': 0.00436046511627907, 'pre': 0.00860832137733142},
# {'task': 'ner-grit', 'model': 'scratch_lr6.25e-5_l6', 'acc': 0.8493557978196233, 'f1': 0.2768729641693811, 'rec': 0.24709302325581395, 'pre': 0.261136712749616},
# {'task': 'ner-grit', 'model': 'babert-bpe-mlm-uncased-128-dup10-5_lr1e-5_l12', 'acc': 0.9447826702534334, 'f1': 0.7574404761904762, 'rec': 0.7398255813953488, 'pre': 0.7485294117647058},
# {'task': 'ner-grit', 'model': 'scratch_lr6.25e-5_l2', 'acc': 0.8231629619142008, 'f1': 0.32967032967032966, 'rec': 0.0436046511627907, 'pre': 0.07702182284980745},
# {'task': 'ner-grit', 'model': 'cartobert_lr1e-5_l12', 'acc': 0.8789466232479116, 'f1': 0.4857142857142857, 'rec': 0.3706395348837209, 'pre': 0.4204451772464963},
# {'task': 'ner-grit', 'model': 'xlm-roberta-base_lr1e-5_l12', 'acc': 0.9520033979895228, 'f1': 0.7836338418862691, 'rec': 0.8212209302325582, 'pre': 0.8019872249822569},
# {'task': 'ner-grit', 'model': 'xlm-mlm-100-1280_lr1e-5_l24', 'acc': 0.9559677190995328, 'f1': 0.7879656160458453, 'rec': 0.7994186046511628, 'pre': 0.7936507936507937},
# {'task': 'ner-grit', 'model': 'babert-bpe-mlm-large-uncased_lr1e-5_l12', 'acc': 0.9453490018405777, 'f1': 0.7209944751381215, 'rec': 0.7587209302325582, 'pre': 0.7393767705382436},
# {'task': 'ner-grit', 'model': 'babert-bpe-mlm-large-uncased-1m_lr1e-5_l12', 'acc': 0.9515786492991647, 'f1': 0.7751060820367751, 'rec': 0.7965116279069767, 'pre': 0.7856630824372759},
# {'task': 'ner-grit', 'model': 'albert-base-uncased-96000_lr1e-5_l12', 'acc': 0.9384114398980603, 'f1': 0.7190201729106628, 'rec': 0.7252906976744186, 'pre': 0.7221418234442837},
# {'task': 'ner-grit', 'model': 'albert-base-uncased-112500_lr1e-5_l12', 'acc': 0.9375619425173439, 'f1': 0.7205240174672489, 'rec': 0.7194767441860465, 'pre': 0.7199999999999999},
# # {'task': 'absa-prosa', 'model': 'bert-base-multilingual-uncased_lr1e-5_l12', 'acc': 0.8759259259259259, 'f1': 0.6306611871181133, 'rec': 0.6053049382428157, 'pre': 0.7301753544775081},
# # {'task': 'absa-prosa', 'model': 'babert-opensubtitle_lr1e-5_l12', 'acc': 0.937962962962963, 'f1': 0.852952335324861, 'rec': 0.8238486390988548, 'pre': 0.889278680386898},
# # {'task': 'absa-prosa', 'model': 'babert-base-512_lr1e-5_l12', 'acc': 0.9611111111111111, 'f1': 0.9176838616543231, 'rec': 0.9187540184088933, 'pre': 0.9166470386511641},
# # {'task': 'absa-prosa', 'model': 'xlm-roberta-large_lr1e-5_l24', 'acc': 0.8777777777777778, 'f1': 0.7222866331081569, 'rec': 0.7097251325033896, 'pre': 0.7418439032855835},
# # {'task': 'absa-prosa', 'model': 'scratch_lr6.25e-5_l4', 'acc': 0.8851851851851852, 'f1': 0.7231836409465563, 'rec': 0.6979175341988112, 'pre': 0.7572776429919288},
# # {'task': 'absa-prosa', 'model': 'scratch_lr6.25e-5_l6', 'acc': 0.887962962962963, 'f1': 0.7523381537096018, 'rec': 0.7343567991756085, 'pre': 0.7953104619602409},
# # {'task': 'absa-prosa', 'model': 'babert-bpe-mlm-uncased-128-dup10-5_lr1e-5_l12', 'acc': 0.9611111111111111, 'f1': 0.9197920124740256, 'rec': 0.9205566808025823, 'pre': 0.9190338190062347},
# # {'task': 'absa-prosa', 'model': 'scratch_lr6.25e-5_l2', 'acc': 0.8851851851851852, 'f1': 0.7253859603907359, 'rec': 0.6981503104238221, 'pre': 0.7597402665850982},
# # {'task': 'absa-prosa', 'model': 'cartobert_lr1e-5_l12', 'acc': 0.8453703703703703, 'f1': 0.5697697803248873, 'rec': 0.5350152630135374, 'pre': 0.6976896263647626},
# # {'task': 'absa-prosa', 'model': 'xlm-roberta-base_lr1e-5_l12', 'acc': 0.862037037037037, 'f1': 0.6228055253811604, 'rec': 0.5901607419079378, 'pre': 0.7244401519327389},
# # {'task': 'absa-prosa', 'model': 'xlm-mlm-100-1280_lr1e-5_l24', 'acc': 0.8712962962962963, 'f1': 0.7034940353088591, 'rec': 0.6939087028603854, 'pre': 0.745897757087392},
# # {'task': 'absa-prosa', 'model': 'babert-bpe-mlm-large-uncased_lr1e-5_l12', 'acc': 0.9648148148148148, 'f1': 0.9278937661345776, 'rec': 0.9234117637914014, 'pre': 0.9324974262941298},
# # {'task': 'absa-prosa', 'model': 'babert-bpe-mlm-large-uncased-1m_lr1e-5_l12', 'acc': 0.9611111111111111, 'f1': 0.928547038737784, 'rec': 0.9316054313897282, 'pre': 0.9261570358260922},
# # {'task': 'absa-prosa', 'model': 'albert-base-uncased-96000_lr1e-5_l12', 'acc': 0.95, 'f1': 0.8949170729888909, 'rec': 0.8703407277609175, 'pre': 0.922945054945055},
# # {'task': 'absa-prosa', 'model': 'albert-base-uncased-112500_lr1e-5_l12', 'acc': 0.9518518518518518, 'f1': 0.8967741881809529, 'rec': 0.8793497367699267, 'pre': 0.9163234352999382},
# {'task': 'emotion-twitter', 'model': 'fasttext-cc-id-300-no-oov-uncased_lr1e-4_l2', 'acc': 0.6318181818181818, 'f1': 0.6474420815199204, 'rec': 0.6450695838814651, 'pre': 0.6665386503908186},
# {'task': 'emotion-twitter', 'model': 'fasttext-cc-id-300-no-oov-uncased_lr1e-4_l6', 'acc': 0.6636363636363637, 'f1': 0.6743040608123556, 'rec': 0.66622954891643, 'pre': 0.6868084929629832},
# {'task': 'emotion-twitter', 'model': 'fasttext-4B-id-300-no-oov-uncased_lr1e-4_l6', 'acc': 0.6772727272727272, 'f1': 0.6846927985145632, 'rec': 0.6812974624385516, 'pre': 0.694475020081571},
# {'task': 'emotion-twitter', 'model': 'fasttext-4B-id-300-no-oov-uncased_lr1e-4_l2', 'acc': 0.7022727272727273, 'f1': 0.7097239930610372, 'rec': 0.7088454701239355, 'pre': 0.7137882530144875},
# {'task': 'emotion-twitter', 'model': 'fasttext-4B-id-300-no-oov-uncased_lr1e-4_l4', 'acc': 0.6886363636363636, 'f1': 0.6923087687734532, 'rec': 0.690845504742782, 'pre': 0.6999734709765686},
# {'task': 'emotion-twitter', 'model': 'babert-bpe-mlm-large-uncased-1100k_lr1e-5_l12', 'acc': 0.7659090909090909, 'f1': 0.7714451699407722, 'rec': 0.7704041750328878, 'pre': 0.7736643529292258},
# {'task': 'emotion-twitter', 'model': 'fasttext-cc-id-300-no-oov-uncased_lr1e-4_l4', 'acc': 0.6454545454545455, 'f1': 0.6535592945416828, 'rec': 0.6519848282905214, 'pre': 0.6608585943094885},
# {'task': 'emotion-twitter', 'model': 'albert-base-uncased-191k_lr1e-5_l12', 'acc': 0.7113636363636363, 'f1': 0.719533483502628, 'rec': 0.7130790002077131, 'pre': 0.7314401535182184},
# {'task': 'term-extraction-airy', 'model': 'fasttext-cc-id-300-no-oov-uncased_lr1e-4_l2', 'acc': 0.8753788337980362, 'f1': 0.6911703456892961, 'rec': 0.8003375934410417, 'pre': 0.7417588557380713},
# {'task': 'term-extraction-airy', 'model': 'bert-base-multilingual-uncased_lr1e-5_l12', 'acc': 0.939629045944963, 'f1': 0.8841834347215723, 'rec': 0.911261152640463, 'pre': 0.8975181094881843},
# {'task': 'term-extraction-airy', 'model': 'fasttext-cc-id-300-no-oov-uncased_lr1e-4_l6', 'acc': 0.8896229846041944, 'f1': 0.7362306368330465, 'rec': 0.8251748251748252, 'pre': 0.7781694144400227},
# {'task': 'term-extraction-airy', 'model': 'babert-opensubtitle_lr1e-5_l12', 'acc': 0.9351436537762153, 'f1': 0.8800755429650614, 'rec': 0.8989631058596576, 'pre': 0.8894190623881666},
# {'task': 'term-extraction-airy', 'model': 'babert-base-512_lr1e-5_l12', 'acc': 0.944599345375197, 'f1': 0.8942917547568711, 'rec': 0.9180130214612974, 'pre': 0.9059971442170396},
# {'task': 'term-extraction-airy', 'model': 'xlm-roberta-large_lr1e-5_l24', 'acc': 0.946781428051885, 'f1': 0.9046834549305719, 'rec': 0.9269351338316856, 'pre': 0.9156741305383517},
# {'task': 'term-extraction-airy', 'model': 'fasttext-4B-id-300-no-oov-uncased_lr1e-4_l6', 'acc': 0.8816220147896715, 'f1': 0.7111158523575848, 'rec': 0.8037135278514589, 'pre': 0.754584559655875},
# {'task': 'term-extraction-airy', 'model': 'scratch_lr6.25e-5_l4', 'acc': 0.888653170081222, 'f1': 0.749832327297116, 'rec': 0.8087774294670846, 'pre': 0.7781902552204176},
# {'task': 'term-extraction-airy', 'model': 'scratch_lr6.25e-5_l6', 'acc': 0.8934416292883985, 'f1': 0.7540654934283805, 'rec': 0.8162527128044369, 'pre': 0.7839277443260768},
# {'task': 'term-extraction-airy', 'model': 'babert-bpe-mlm-uncased-128-dup10-5_lr1e-5_l12', 'acc': 0.9452054794520548, 'f1': 0.8951821386603995, 'rec': 0.9184952978056427, 'pre': 0.906688883599143},
# {'task': 'term-extraction-airy', 'model': 'scratch_lr6.25e-5_l2', 'acc': 0.8826524427203297, 'f1': 0.7280604310153299, 'rec': 0.7902097902097902, 'pre': 0.7578630897317298},
# {'task': 'term-extraction-airy', 'model': 'cartobert_lr1e-5_l12', 'acc': 0.9252636683234331, 'f1': 0.8607838422697763, 'rec': 0.8632746563781046, 'pre': 0.8620274500361184},
# {'task': 'term-extraction-airy', 'model': 'fasttext-4B-id-300-no-oov-uncased_lr1e-4_l2', 'acc': 0.8731361377136623, 'f1': 0.6880613362541074, 'rec': 0.7574149987943092, 'pre': 0.7210743801652894},
# {'task': 'term-extraction-airy', 'model': 'xlm-roberta-base_lr1e-5_l12', 'acc': 0.9439932112983392, 'f1': 0.8946261682242991, 'rec': 0.9233180612490958, 'pre': 0.9087456983505401},
# {'task': 'term-extraction-airy', 'model': 'xlm-mlm-100-1280_lr1e-5_l24', 'acc': 0.9444781185598254, 'f1': 0.8919738684087728, 'rec': 0.9218712322160598, 'pre': 0.9066761532076366},
# {'task': 'term-extraction-airy', 'model': 'babert-bpe-mlm-large-uncased_lr1e-5_l12', 'acc': 0.9474481755364287, 'f1': 0.8968494749124855, 'rec': 0.9266939956595129, 'pre': 0.9115275142314991},
# {'task': 'term-extraction-airy', 'model': 'babert-bpe-mlm-large-uncased-1m_lr1e-5_l12', 'acc': 0.9478724693902291, 'f1': 0.9049195837275308, 'rec': 0.9225946467325777, 'pre': 0.9136716417910448},
# {'task': 'term-extraction-airy', 'model': 'fasttext-4B-id-300-no-oov-uncased_lr1e-4_l4', 'acc': 0.8779245969208389, 'f1': 0.7025555090071219, 'rec': 0.8087774294670846, 'pre': 0.751933639726488},
# {'task': 'term-extraction-airy', 'model': 'babert-bpe-mlm-large-uncased-1100k_lr1e-5_l12', 'acc': 0.9494484179900594, 'f1': 0.9087038789025544, 'rec': 0.9264528574873403, 'pre': 0.9174925373134328},
# {'task': 'term-extraction-airy', 'model': 'fasttext-cc-id-300-no-oov-uncased_lr1e-4_l4', 'acc': 0.8866529276275912, 'f1': 0.7248936170212766, 'rec': 0.8215577525922354, 'pre': 0.7702045891262576},
# {'task': 'term-extraction-airy', 'model': 'albert-base-uncased-96000_lr1e-5_l12', 'acc': 0.9399927263910777, 'f1': 0.886985486557221, 'rec': 0.8989631058596576, 'pre': 0.8929341317365269},
# {'task': 'term-extraction-airy', 'model': 'albert-base-uncased-191k_lr1e-5_l12', 'acc': 0.9380530973451328, 'f1': 0.8955657862854374, 'rec': 0.8912466843501327, 'pre': 0.8934010152284265},
# {'task': 'term-extraction-airy', 'model': 'albert-base-uncased-112500_lr1e-5_l12', 'acc': 0.9390229118681053, 'f1': 0.8917365269461078, 'rec': 0.8977574149987944, 'pre': 0.894736842105263},
# {'task': 'doc-sentiment-prosa', 'model': 'fasttext-cc-id-300-no-oov-uncased_lr1e-4_l2', 'acc': 0.798, 'f1': 0.7670700170098602, 'rec': 0.7632432926550573, 'pre': 0.7749126330465117},
# {'task': 'doc-sentiment-prosa', 'model': 'bert-base-multilingual-uncased_lr1e-5_l12', 'acc': 0.88, 'f1': 0.8413969690565436, 'rec': 0.8249548653960419, 'pre': 0.8768647717484926},
# {'task': 'doc-sentiment-prosa', 'model': 'fasttext-cc-id-300-no-oov-uncased_lr1e-4_l6', 'acc': 0.816, 'f1': 0.7884143405122015, 'rec': 0.7733271630330454, 'pre': 0.8190044735109113},
# {'task': 'doc-sentiment-prosa', 'model': 'babert-opensubtitle_lr1e-5_l12', 'acc': 0.868, 'f1': 0.8323185385445452, 'rec': 0.8174305269893504, 'pre': 0.8621238608022749},
# {'task': 'doc-sentiment-prosa', 'model': 'babert-base-512_lr1e-5_l12', 'acc': 0.93, 'f1': 0.9089927399868921, 'rec': 0.8979243795420265, 'pre': 0.9246343416242905},
# {'task': 'doc-sentiment-prosa', 'model': 'xlm-roberta-large_lr1e-5_l24', 'acc': 0.94, 'f1': 0.923465800494001, 'rec': 0.9125559897618721, 'pre': 0.93886155073813},
# {'task': 'doc-sentiment-prosa', 'model': 'fasttext-4B-id-300-no-oov-uncased_lr1e-4_l6', 'acc': 0.854, 'f1': 0.8306889194379359, 'rec': 0.8170448832213538, 'pre': 0.8548546926303883},
# {'task': 'doc-sentiment-prosa', 'model': 'scratch_lr6.25e-5_l4', 'acc': 0.732, 'f1': 0.6805295919382729, 'rec': 0.6684292015174368, 'pre': 0.7225756969454449},
# {'task': 'doc-sentiment-prosa', 'model': 'scratch_lr6.25e-5_l6', 'acc': 0.704, 'f1': 0.661719455302218, 'rec': 0.6551744823803647, 'pre': 0.7060584948023009},
# {'task': 'doc-sentiment-prosa', 'model': 'babert-bpe-mlm-uncased-128-dup10-5_lr1e-5_l12', 'acc': 0.92, 'f1': 0.8962038475097632, 'rec': 0.8834184606243429, 'pre': 0.9173194442907774},
# {'task': 'doc-sentiment-prosa', 'model': 'scratch_lr6.25e-5_l2', 'acc': 0.736, 'f1': 0.6975865918680123, 'rec': 0.6873086064262535, 'pre': 0.7351670370227282},
# {'task': 'doc-sentiment-prosa', 'model': 'cartobert_lr1e-5_l12', 'acc': 0.814, 'f1': 0.7697235559667566, 'rec': 0.7522453037158918, 'pre': 0.8239834462131613},
# {'task': 'doc-sentiment-prosa', 'model': 'fasttext-4B-id-300-no-oov-uncased_lr1e-4_l2', 'acc': 0.854, 'f1': 0.8362938415728527, 'rec': 0.8279714566479273, 'pre': 0.8492724951737527},
# {'task': 'doc-sentiment-prosa', 'model': 'xlm-roberta-base_lr1e-5_l12', 'acc': 0.95, 'f1': 0.9355928176257132, 'rec': 0.9205688102746926, 'pre': 0.9580765175349112},
# {'task': 'doc-sentiment-prosa', 'model': 'xlm-mlm-100-1280_lr1e-5_l24', 'acc': 0.894, 'f1': 0.863286442931254, 'rec': 0.8450083413318707, 'pre': 0.9014803195530905},
# {'task': 'doc-sentiment-prosa', 'model': 'babert-bpe-mlm-large-uncased_lr1e-5_l12', 'acc': 0.958, 'f1': 0.9457101131799927, 'rec': 0.9357203254262078, 'pre': 0.958899728827265},
# {'task': 'doc-sentiment-prosa', 'model': 'babert-bpe-mlm-large-uncased-1m_lr1e-5_l12', 'acc': 0.95, 'f1': 0.934979212885163, 'rec': 0.9227855477855478, 'pre': 0.9523832945512517},
# {'task': 'doc-sentiment-prosa', 'model': 'fasttext-4B-id-300-no-oov-uncased_lr1e-4_l4', 'acc': 0.852, 'f1': 0.8213106006704858, 'rec': 0.8046100141688376, 'pre': 0.8562633500098332},
# {'task': 'doc-sentiment-prosa', 'model': 'babert-bpe-mlm-large-uncased-1100k_lr1e-5_l12', 'acc': 0.946, 'f1': 0.9308052575600184, 'rec': 0.9195804195804196, 'pre': 0.9469850145712214},
# {'task': 'doc-sentiment-prosa', 'model': 'fasttext-cc-id-300-no-oov-uncased_lr1e-4_l4', 'acc': 0.8, 'f1': 0.769166975494454, 'rec': 0.7561045980163628, 'pre': 0.795664616548763},
# {'task': 'doc-sentiment-prosa', 'model': 'albert-base-uncased-96000_lr1e-5_l12', 'acc': 0.93, 'f1': 0.908014642460799, 'rec': 0.891399858311623, 'pre': 0.9360872508640776},
# {'task': 'doc-sentiment-prosa', 'model': 'albert-base-uncased-191k_lr1e-5_l12', 'acc': 0.92, 'f1': 0.9011770612036608, 'rec': 0.8942821883998354, 'pre': 0.9106066667010033},
# {'task': 'doc-sentiment-prosa', 'model': 'albert-base-uncased-112500_lr1e-5_l12', 'acc': 0.928, 'f1': 0.9084541140489831, 'rec': 0.9073112345171168, 'pre': 0.9111269036085856},
# {'task': 'entailment-ui', 'model': 'fasttext-cc-id-300-no-oov-uncased_lr1e-4_l2', 'acc': 0.66, 'f1': 0.6510673234811166, 'rec': 0.6565783942833123, 'pre': 0.6505050505050505},
# {'task': 'entailment-ui', 'model': 'fasttext-cc-id-300-no-oov-uncased_lr1e-4_l6', 'acc': 0.69, 'f1': 0.6112852664576802, 'rec': 0.6210592686002523, 'pre': 0.7142857142857143},
# {'task': 'entailment-ui', 'model': 'fasttext-4B-id-300-no-oov-uncased_lr1e-4_l6', 'acc': 0.61, 'f1': 0.6032956972841013, 'rec': 0.6109709962168979, 'pre': 0.6057692307692308},
# {'task': 'entailment-ui', 'model': 'fasttext-4B-id-300-no-oov-uncased_lr1e-4_l2', 'acc': 0.59, 'f1': 0.5710848415106182, 'rec': 0.5714585960487599, 'pre': 0.5708333333333333},
# {'task': 'entailment-ui', 'model': 'fasttext-4B-id-300-no-oov-uncased_lr1e-4_l4', 'acc': 0.62, 'f1': 0.6041666666666667, 'rec': 0.605296343001261, 'pre': 0.6035551880942538},
# {'task': 'entailment-ui', 'model': 'babert-bpe-mlm-large-uncased-1100k_lr1e-5_l12', 'acc': 0.81, 'f1': 0.7925537722458784, 'rec': 0.7841530054644809, 'pre': 0.8106617647058824},
# {'task': 'entailment-ui', 'model': 'fasttext-cc-id-300-no-oov-uncased_lr1e-4_l4', 'acc': 0.68, 'f1': 0.6736026111791105, 'rec': 0.6822194199243379, 'pre': 0.6740264953833802},
# {'task': 'entailment-ui', 'model': 'albert-base-uncased-191k_lr1e-5_l12', 'acc': 0.83, 'f1': 0.8030355694589271, 'rec': 0.7866750735603194, 'pre': 0.8739035087719298},
# {'task': 'keyword-extraction-prosa', 'model': 'fasttext-cc-id-300-no-oov-uncased_lr1e-4_l2', 'acc': 0.7900328864153807, 'f1': 0.47771173848439824, 'rec': 0.6541200406917599, 'pre': 0.5521683125805067},
# {'task': 'keyword-extraction-prosa', 'model': 'bert-base-multilingual-uncased_lr1e-5_l12', 'acc': 0.8234252466481153, 'f1': 0.5749158249158249, 'rec': 0.6948118006103764, 'pre': 0.6292031321971442},
# {'task': 'keyword-extraction-prosa', 'model': 'fasttext-cc-id-300-no-oov-uncased_lr1e-4_l6', 'acc': 0.7945863900834809, 'f1': 0.5, 'rec': 0.6703967446592065, 'pre': 0.5727944372012168},
# {'task': 'keyword-extraction-prosa', 'model': 'babert-opensubtitle_lr1e-5_l12', 'acc': 0.8279787503162155, 'f1': 0.5733333333333334, 'rec': 0.6998982706002035, 'pre': 0.6303252404947319},
# {'task': 'keyword-extraction-prosa', 'model': 'babert-base-512_lr1e-5_l12', 'acc': 0.8535289653427777, 'f1': 0.6451048951048951, 'rec': 0.7507629704984741, 'pre': 0.693935119887165},
# {'task': 'keyword-extraction-prosa', 'model': 'xlm-roberta-large_lr1e-5_l24', 'acc': 0.8537819377687832, 'f1': 0.6460251046025105, 'rec': 0.785350966429298, 'pre': 0.7089072543617998},
# {'task': 'keyword-extraction-prosa', 'model': 'fasttext-4B-id-300-no-oov-uncased_lr1e-4_l6', 'acc': 0.7391854287882621, 'f1': 0.476409666283084, 'rec': 0.4211597151576806, 'pre': 0.4470842332613391},
# {'task': 'keyword-extraction-prosa', 'model': 'scratch_lr6.25e-5_l4', 'acc': 0.7616999747027574, 'f1': 0.43894899536321486, 'rec': 0.5778229908443541, 'pre': 0.4989020641194555},
# {'task': 'keyword-extraction-prosa', 'model': 'scratch_lr6.25e-5_l6', 'acc': 0.7503162155325069, 'f1': 0.4303977272727273, 'rec': 0.6164801627670397, 'pre': 0.506900878293601},
# {'task': 'keyword-extraction-prosa', 'model': 'babert-bpe-mlm-uncased-128-dup10-5_lr1e-5_l12', 'acc': 0.8535289653427777, 'f1': 0.6404109589041096, 'rec': 0.7609359104781281, 'pre': 0.695490469549047},
# {'task': 'keyword-extraction-prosa', 'model': 'scratch_lr6.25e-5_l2', 'acc': 0.7568934986086516, 'f1': 0.4405487804878049, 'rec': 0.5879959308240081, 'pre': 0.5037037037037037},
# {'task': 'keyword-extraction-prosa', 'model': 'cartobert_lr1e-5_l12', 'acc': 0.7998988110295978, 'f1': 0.5190097259062776, 'rec': 0.5971515768056969, 'pre': 0.5553453169347209},
# {'task': 'keyword-extraction-prosa', 'model': 'fasttext-4B-id-300-no-oov-uncased_lr1e-4_l2', 'acc': 0.7920566658234253, 'f1': 0.4844765342960289, 'rec': 0.6826042726347915, 'pre': 0.566722972972973},
# {'task': 'keyword-extraction-prosa', 'model': 'xlm-roberta-base_lr1e-5_l12', 'acc': 0.8449279028585884, 'f1': 0.6252019386106623, 'rec': 0.7873855544252288, 'pre': 0.6969833408374606},
# {'task': 'keyword-extraction-prosa', 'model': 'xlm-mlm-100-1280_lr1e-5_l24', 'acc': 0.8289906400202378, 'f1': 0.583533173461231, 'rec': 0.7426246185147508, 'pre': 0.6535362578334826},
# {'task': 'keyword-extraction-prosa', 'model': 'babert-bpe-mlm-large-uncased_lr1e-5_l12', 'acc': 0.847457627118644, 'f1': 0.612565445026178, 'rec': 0.7141403865717192, 'pre': 0.6594645373414749},
# {'task': 'keyword-extraction-prosa', 'model': 'babert-bpe-mlm-large-uncased-1m_lr1e-5_l12', 'acc': 0.855046799898811, 'f1': 0.63881636205396, 'rec': 0.7466937945066124, 'pre': 0.6885553470919324},
# {'task': 'keyword-extraction-prosa', 'model': 'fasttext-4B-id-300-no-oov-uncased_lr1e-4_l4', 'acc': 0.7814318239311915, 'f1': 0.48755490483162517, 'rec': 0.6775178026449644, 'pre': 0.5670498084291188},
# {'task': 'keyword-extraction-prosa', 'model': 'babert-bpe-mlm-large-uncased-1100k_lr1e-5_l12', 'acc': 0.8537819377687832, 'f1': 0.6434316353887399, 'rec': 0.7324516785350966, 'pre': 0.6850618458610847},
# {'task': 'keyword-extraction-prosa', 'model': 'fasttext-cc-id-300-no-oov-uncased_lr1e-4_l4', 'acc': 0.7918036933974196, 'f1': 0.48301329394387, 'rec': 0.6653102746693794, 'pre': 0.5596919127086007},
# {'task': 'keyword-extraction-prosa', 'model': 'albert-base-uncased-96000_lr1e-5_l12', 'acc': 0.8472046546926385, 'f1': 0.62, 'rec': 0.7568667344862665, 'pre': 0.6816307833256985},
# {'task': 'keyword-extraction-prosa', 'model': 'albert-base-uncased-191k_lr1e-5_l12', 'acc': 0.8466987098406273, 'f1': 0.5993322203672788, 'rec': 0.7304170905391658, 'pre': 0.6584135717560752},
# {'task': 'keyword-extraction-prosa', 'model': 'albert-base-uncased-112500_lr1e-5_l12', 'acc': 0.8507462686567164, 'f1': 0.633304572907679, 'rec': 0.7466937945066124, 'pre': 0.6853408029878618},
# # {'task': 'absa-airy', 'model': 'fasttext-cc-id-300-no-oov-uncased_lr1e-4_l2', 'acc': 0.9370629370629371, 'f1': 0.7805447374135975, 'rec': 0.7483109229521, 'pre': 0.8472575746332293},
# # {'task': 'absa-airy', 'model': 'bert-base-multilingual-uncased_lr1e-5_l12', 'acc': 0.9391608391608391, 'f1': 0.824018343364861, 'rec': 0.7916804826235118, 'pre': 0.8681412793408105},
# # {'task': 'absa-airy', 'model': 'fasttext-cc-id-300-no-oov-uncased_lr1e-4_l6', 'acc': 0.9468531468531468, 'f1': 0.8500759915256445, 'rec': 0.8434606313800875, 'pre': 0.857234813164378},
# # {'task': 'absa-airy', 'model': 'babert-opensubtitle_lr1e-5_l12', 'acc': 0.9552447552447553, 'f1': 0.8700052764591732, 'rec': 0.8427494324744638, 'pre': 0.9058887489362871},
# # {'task': 'absa-airy', 'model': 'babert-base-512_lr1e-5_l12', 'acc': 0.9636363636363636, 'f1': 0.9076282241857615, 'rec': 0.8896759414126754, 'pre': 0.9283009532483013},
# # {'task': 'absa-airy', 'model': 'xlm-roberta-large_lr1e-5_l24', 'acc': 0.9265734265734266, 'f1': 0.7975756806742722, 'rec': 0.7501018128888922, 'pre': 0.8773516948513841},
# # {'task': 'absa-airy', 'model': 'fasttext-4B-id-300-no-oov-uncased_lr1e-4_l6', 'acc': 0.9545454545454546, 'f1': 0.8619675063350956, 'rec': 0.8476083492593367, 'pre': 0.8789601432662043},
# # {'task': 'absa-airy', 'model': 'scratch_lr6.25e-5_l4', 'acc': 0.9395104895104895, 'f1': 0.8431632673527124, 'rec': 0.8356001568021364, 'pre': 0.8510756001425285},
# # {'task': 'absa-airy', 'model': 'scratch_lr6.25e-5_l6', 'acc': 0.9444055944055944, 'f1': 0.8502976483627988, 'rec': 0.8247451554524022, 'pre': 0.8842763623513522},
# # {'task': 'absa-airy', 'model': 'babert-bpe-mlm-uncased-128-dup10-5_lr1e-5_l12', 'acc': 0.9618881118881119, 'f1': 0.9031426026692376, 'rec': 0.8873406135932319, 'pre': 0.9216262980044823},
# # {'task': 'absa-airy', 'model': 'scratch_lr6.25e-5_l2', 'acc': 0.9332167832167833, 'f1': 0.8186697556375372, 'rec': 0.8043480738620777, 'pre': 0.8343895221322665},
# # {'task': 'absa-airy', 'model': 'cartobert_lr1e-5_l12', 'acc': 0.9262237762237763, 'f1': 0.7638970202726544, 'rec': 0.7155130958840505, 'pre': 0.8673279610198872},
# # {'task': 'absa-airy', 'model': 'fasttext-4B-id-300-no-oov-uncased_lr1e-4_l2', 'acc': 0.9437062937062937, 'f1': 0.8021028245971197, 'rec': 0.7697288570269647, 'pre': 0.864213660226905},
# # {'task': 'absa-airy', 'model': 'xlm-roberta-base_lr1e-5_l12', 'acc': 0.926923076923077, 'f1': 0.7875740014575415, 'rec': 0.7467079019043159, 'pre': 0.8497396403346569},
# # {'task': 'absa-airy', 'model': 'babert-bpe-mlm-large-uncased_lr1e-5_l12', 'acc': 0.9660839160839161, 'f1': 0.9155846130428739, 'rec': 0.9051765753665609, 'pre': 0.9265333951480547},
# # {'task': 'absa-airy', 'model': 'babert-bpe-mlm-large-uncased-1m_lr1e-5_l12', 'acc': 0.9692307692307692, 'f1': 0.9298405121414856, 'rec': 0.9248251270287217, 'pre': 0.9349722324605084},
# # {'task': 'absa-airy', 'model': 'fasttext-4B-id-300-no-oov-uncased_lr1e-4_l4', 'acc': 0.9513986013986014, 'f1': 0.8588408235924092, 'rec': 0.8446928067604617, 'pre': 0.8771781810536087},
# # {'task': 'absa-airy', 'model': 'babert-bpe-mlm-large-uncased-1100k_lr1e-5_l12', 'acc': 0.9678321678321679, 'f1': 0.924771022320111, 'rec': 0.9203818250142216, 'pre': 0.929253489491756},
# # {'task': 'absa-airy', 'model': 'fasttext-cc-id-300-no-oov-uncased_lr1e-4_l4', 'acc': 0.9520979020979021, 'f1': 0.8531965564760317, 'rec': 0.8408009810509832, 'pre': 0.8682219119755227},
# # {'task': 'absa-airy', 'model': 'albert-base-uncased-96000_lr1e-5_l12', 'acc': 0.9566433566433566, 'f1': 0.8900428824305896, 'rec': 0.8597490875695644, 'pre': 0.9280235671630658},
# # {'task': 'absa-airy', 'model': 'albert-base-uncased-191k_lr1e-5_l12', 'acc': 0.9552447552447553, 'f1': 0.8767121150258529, 'rec': 0.8417876183655782, 'pre': 0.9242510188997697},
# # {'task': 'absa-airy', 'model': 'albert-base-uncased-112500_lr1e-5_l12', 'acc': 0.9545454545454546, 'f1': 0.8807341617579502, 'rec': 0.8521196819556742, 'pre': 0.9167367070151652},
# {'task': 'ner-grit', 'model': 'fasttext-cc-id-300-no-oov-uncased_lr1e-4_l2', 'acc': 0.8742743876539714, 'f1': 0.321353065539112, 'rec': 0.4418604651162791, 'pre': 0.37209302325581395},
# {'task': 'ner-grit', 'model': 'fasttext-cc-id-300-no-oov-uncased_lr1e-4_l6', 'acc': 0.8640804190853745, 'f1': 0.3760539629005059, 'rec': 0.3241279069767442, 'pre': 0.3481654957064793},
# {'task': 'ner-grit', 'model': 'fasttext-4B-id-300-no-oov-uncased_lr1e-4_l6', 'acc': 0.894237576100807, 'f1': 0.41468926553672314, 'rec': 0.5334302325581395, 'pre': 0.46662428480610296},
# {'task': 'ner-grit', 'model': 'fasttext-4B-id-300-no-oov-uncased_lr1e-4_l2', 'acc': 0.8805040351125584, 'f1': 0.3392018779342723, 'rec': 0.42005813953488375, 'pre': 0.37532467532467534},
# {'task': 'ner-grit', 'model': 'fasttext-4B-id-300-no-oov-uncased_lr1e-4_l4', 'acc': 0.8908395865779414, 'f1': 0.39931350114416475, 'rec': 0.5072674418604651, 'pre': 0.44686299615877084},
# {'task': 'ner-grit', 'model': 'babert-bpe-mlm-large-uncased-1100k_lr1e-5_l12', 'acc': 0.9488885742602293, 'f1': 0.7677053824362606, 'rec': 0.7877906976744186, 'pre': 0.7776183644189383},
# {'task': 'ner-grit', 'model': 'fasttext-cc-id-300-no-oov-uncased_lr1e-4_l4', 'acc': 0.8843267733257822, 'f1': 0.36536430834213307, 'rec': 0.502906976744186, 'pre': 0.4232415902140673},
# {'task': 'ner-grit', 'model': 'albert-base-uncased-191k_lr1e-5_l12', 'acc': 0.9375619425173439, 'f1': 0.7469512195121951, 'rec': 0.7122093023255814, 'pre': 0.7291666666666667},
# {'task': 'pos-prosa', 'model': 'fasttext-cc-id-300-no-oov-uncased_lr1e-4_l2', 'acc': 0.9547422085417469, 'f1': 0.952836808340016, 'rec': 0.952836808340016, 'pre': 0.952836808340016},
# {'task': 'pos-prosa', 'model': 'bert-base-multilingual-uncased_lr1e-5_l12', 'acc': 0.9592631781454406, 'f1': 0.9575481154771451, 'rec': 0.9575481154771451, 'pre': 0.9575481154771451},
# {'task': 'pos-prosa', 'model': 'fasttext-cc-id-300-no-oov-uncased_lr1e-4_l6', 'acc': 0.9574355521354367, 'f1': 0.9556435445068163, 'rec': 0.9556435445068163, 'pre': 0.9556435445068163},
# {'task': 'pos-prosa', 'model': 'babert-opensubtitle_lr1e-5_l12', 'acc': 0.9527702962677953, 'f1': 0.9507818765036087, 'rec': 0.9507818765036087, 'pre': 0.9507818765036087},
# {'task': 'pos-prosa', 'model': 'babert-base-512_lr1e-5_l12', 'acc': 0.9645536744901886, 'f1': 0.96306134723336, 'rec': 0.96306134723336, 'pre': 0.96306134723336},
# {'task': 'pos-prosa', 'model': 'xlm-roberta-large_lr1e-5_l24', 'acc': 0.9676317814544055, 'f1': 0.9662690457097033, 'rec': 0.9662690457097033, 'pre': 0.9662690457097033},
# {'task': 'pos-prosa', 'model': 'scratch_lr6.25e-5_l4', 'acc': 0.9143901500577145, 'f1': 0.9107858861267041, 'rec': 0.9107858861267041, 'pre': 0.9107858861267041},
# {'task': 'pos-prosa', 'model': 'scratch_lr6.25e-5_l6', 'acc': 0.9141015775298191, 'f1': 0.910485164394547, 'rec': 0.910485164394547, 'pre': 0.910485164394547},
# {'task': 'pos-prosa', 'model': 'babert-bpe-mlm-uncased-128-dup10-5_lr1e-5_l12', 'acc': 0.9641208156983455, 'f1': 0.9626102646351243, 'rec': 0.9626102646351243, 'pre': 0.9626102646351243},
# {'task': 'pos-prosa', 'model': 'scratch_lr6.25e-5_l2', 'acc': 0.9138611004232398, 'f1': 0.9102345629510826, 'rec': 0.9102345629510826, 'pre': 0.9102345629510826},
# {'task': 'pos-prosa', 'model': 'cartobert_lr1e-5_l12', 'acc': 0.9203539823008849, 'f1': 0.9170008019246191, 'rec': 0.9170008019246191, 'pre': 0.9170008019246191},
# {'task': 'pos-prosa', 'model': 'fasttext-4B-id-300-no-oov-uncased_lr1e-4_l2', 'acc': 0.9583012697191228, 'f1': 0.9565457097032879, 'rec': 0.9565457097032879, 'pre': 0.956545709703288},
# {'task': 'pos-prosa', 'model': 'xlm-roberta-base_lr1e-5_l12', 'acc': 0.9606579453636014, 'f1': 0.9590016038492382, 'rec': 0.9590016038492382, 'pre': 0.9590016038492382},
# {'task': 'pos-prosa', 'model': 'xlm-mlm-100-1280_lr1e-5_l24', 'acc': 0.9682089265101962, 'f1': 0.9668704891740176, 'rec': 0.9668704891740176, 'pre': 0.9668704891740176},
# {'task': 'pos-prosa', 'model': 'babert-bpe-mlm-large-uncased_lr1e-5_l12', 'acc': 0.9649384378607156, 'f1': 0.963462309542903, 'rec': 0.963462309542903, 'pre': 0.963462309542903},
# {'task': 'pos-prosa', 'model': 'babert-bpe-mlm-large-uncased-1m_lr1e-5_l12', 'acc': 0.9646979607541362, 'f1': 0.9632117080994387, 'rec': 0.9632117080994387, 'pre': 0.9632117080994387},
# {'task': 'pos-prosa', 'model': 'fasttext-4B-id-300-no-oov-uncased_lr1e-4_l4', 'acc': 0.9602731819930742, 'f1': 0.9586006415396953, 'rec': 0.9586006415396953, 'pre': 0.9586006415396953},
# {'task': 'pos-prosa', 'model': 'babert-bpe-mlm-large-uncased-1100k_lr1e-5_l12', 'acc': 0.9646979607541362, 'f1': 0.9632117080994387, 'rec': 0.9632117080994387, 'pre': 0.9632117080994387},
# {'task': 'pos-prosa', 'model': 'fasttext-cc-id-300-no-oov-uncased_lr1e-4_l4', 'acc': 0.9567141208156984, 'f1': 0.9548917401764234, 'rec': 0.9548917401764234, 'pre': 0.9548917401764234},
# {'task': 'pos-prosa', 'model': 'albert-base-uncased-96000_lr1e-5_l12', 'acc': 0.9451231242785687, 'f1': 0.9428127506014434, 'rec': 0.9428127506014434, 'pre': 0.9428127506014434},
# {'task': 'pos-prosa', 'model': 'albert-base-uncased-191k_lr1e-5_l12', 'acc': 0.9415640631011928, 'f1': 0.9391038492381716, 'rec': 0.9391038492381716, 'pre': 0.9391038492381716},
# {'task': 'pos-prosa', 'model': 'albert-base-uncased-112500_lr1e-5_l12', 'acc': 0.944978838014621, 'f1': 0.9426623897353649, 'rec': 0.9426623897353649, 'pre': 0.9426623897353649},
# # {'task': 'absa-prosa', 'model': 'fasttext-cc-id-300-no-oov-uncased_lr1e-4_l2', 'acc': 0.9212962962962963, 'f1': 0.7882963159241282, 'rec': 0.7699969629434685, 'pre': 0.8091999668756084},
# # {'task': 'absa-prosa', 'model': 'fasttext-cc-id-300-no-oov-uncased_lr1e-4_l6', 'acc': 0.9342592592592592, 'f1': 0.8261821573176835, 'rec': 0.8063394807355118, 'pre': 0.8499428794756937},
# # {'task': 'absa-prosa', 'model': 'fasttext-4B-id-300-no-oov-uncased_lr1e-4_l6', 'acc': 0.9324074074074075, 'f1': 0.8195852126661308, 'rec': 0.8004269368592061, 'pre': 0.8447444780504424},
# # {'task': 'absa-prosa', 'model': 'fasttext-4B-id-300-no-oov-uncased_lr1e-4_l2', 'acc': 0.9324074074074075, 'f1': 0.8142923633032911, 'rec': 0.7968216120718278, 'pre': 0.8360526780202345},
# # {'task': 'absa-prosa', 'model': 'fasttext-4B-id-300-no-oov-uncased_lr1e-4_l4', 'acc': 0.9425925925925925, 'f1': 0.8547765608554189, 'rec': 0.8435507799183383, 'pre': 0.8668043505647026},
# # {'task': 'absa-prosa', 'model': 'babert-bpe-mlm-large-uncased-1100k_lr1e-5_l12', 'acc': 0.9638888888888889, 'f1': 0.9340299234665431, 'rec': 0.9371666594487992, 'pre': 0.9309968339380105},
# # {'task': 'absa-prosa', 'model': 'fasttext-cc-id-300-no-oov-uncased_lr1e-4_l4', 'acc': 0.9231481481481482, 'f1': 0.790174166918353, 'rec': 0.7728563488917243, 'pre': 0.8139351225080301},
# # {'task': 'absa-prosa', 'model': 'albert-base-uncased-191k_lr1e-5_l12', 'acc': 0.9342592592592592, 'f1': 0.8470998246095364, 'rec': 0.8045368183418226, 'pre': 0.9071098571098571},
# # {'task': 'absa-prosa', 'model': 'albert-base-wwmlm-512_lr1e-5_l12', 'acc': 0.9425925925925925, 'f1': 0.876344590348324, 'rec': 0.8453534423120272, 'pre': 0.9136922950462282}
# # {'task': 'absa-prosa', 'model': 'albert-large-wwmlm-128_lr1e-5_l12', 'acc': 0.9527777777777777, 'f1': 0.902956670061033, 'rec': 0.9047012469143617, 'pre': 0.9012381138465546}
# # {'task': 'absa-prosa', 'model': 'babert-bpe-mlm-large-512_lr1e-5_l24', 'acc': 0.9648148148148148, 'f1': 0.935764937784129, 'rec': 0.9272498648037906, 'pre': 0.9451353456518873}
# {'task': 'ner-prosa', 'model': 'fasttext-cc-id-300-no-oov-uncased_lr1e-4_l2', 'acc': 0.9268468641785302, 'f1': 0.516636690647482, 'rec': 0.6473239436619719, 'pre': 0.5746436609152289},
# {'task': 'ner-prosa', 'model': 'bert-base-multilingual-uncased_lr1e-5_l12', 'acc': 0.9597922277799154, 'f1': 0.8075385494003426, 'rec': 0.7966197183098591, 'pre': 0.8020419739081112},
# {'task': 'ner-prosa', 'model': 'fasttext-cc-id-300-no-oov-uncased_lr1e-4_l6', 'acc': 0.9292035398230089, 'f1': 0.5220919747520288, 'rec': 0.652394366197183, 'pre': 0.5800150262960181},
# {'task': 'ner-prosa', 'model': 'babert-opensubtitle_lr1e-5_l12', 'acc': 0.9460850327048865, 'f1': 0.6995637949836423, 'rec': 0.7228169014084507, 'pre': 0.7110002770850651},
# {'task': 'ner-prosa', 'model': 'babert-base-512_lr1e-5_l12', 'acc': 0.9628703347441323, 'f1': 0.8178228990411731, 'rec': 0.8169014084507042, 'pre': 0.8173618940248026},
# {'task': 'ner-prosa', 'model': 'xlm-roberta-large_lr1e-5_l24', 'acc': 0.9657079646017699, 'f1': 0.8384230982787341, 'rec': 0.8507042253521127, 'pre': 0.8445190156599552},
# {'task': 'ner-prosa', 'model': 'fasttext-4B-id-300-no-oov-uncased_lr1e-4_l6', 'acc': 0.9356964217006541, 'f1': 0.5478024467603081, 'rec': 0.6811267605633803, 'pre': 0.6072325464590658},
# {'task': 'ner-prosa', 'model': 'scratch_lr6.25e-5_l4', 'acc': 0.8606194690265486, 'f1': 0.3076923076923077, 'rec': 0.0022535211267605635, 'pre': 0.0044742729306487695},
# {'task': 'ner-prosa', 'model': 'scratch_lr6.25e-5_l6', 'acc': 0.8621104270873413, 'f1': 0.5660377358490566, 'rec': 0.016901408450704224, 'pre': 0.03282275711159737},
# {'task': 'ner-prosa', 'model': 'babert-bpe-mlm-uncased-128-dup10-5_lr1e-5_l12', 'acc': 0.9643131973836091, 'f1': 0.8154192459200901, 'rec': 0.8163380281690141, 'pre': 0.8158783783783785},
# {'task': 'ner-prosa', 'model': 'scratch_lr6.25e-5_l2', 'acc': 0.8610523278183917, 'f1': 0.47619047619047616, 'rec': 0.005633802816901409, 'pre': 0.011135857461024499},
# {'task': 'ner-prosa', 'model': 'cartobert_lr1e-5_l12', 'acc': 0.9168911119661408, 'f1': 0.5556986477784932, 'rec': 0.48619718309859156, 'pre': 0.5186298076923077},
# {'task': 'ner-prosa', 'model': 'fasttext-4B-id-300-no-oov-uncased_lr1e-4_l2', 'acc': 0.93223355136591, 'f1': 0.5165683159328189, 'rec': 0.6411267605633802, 'pre': 0.5721468074409252},
# {'task': 'ner-prosa', 'model': 'xlm-roberta-base_lr1e-5_l12', 'acc': 0.9531550596383225, 'f1': 0.8105781057810578, 'rec': 0.7425352112676057, 'pre': 0.7750661570126435},
# {'task': 'ner-prosa', 'model': 'xlm-mlm-100-1280_lr1e-5_l24', 'acc': 0.9619084263178146, 'f1': 0.8042880703683343, 'rec': 0.824225352112676, 'pre': 0.8141346688925987},
# {'task': 'ner-prosa', 'model': 'babert-bpe-mlm-large-uncased_lr1e-5_l12', 'acc': 0.9667179684494036, 'f1': 0.8314917127071824, 'rec': 0.847887323943662, 'pre': 0.8396094839609485},
# {'task': 'ner-prosa', 'model': 'babert-bpe-mlm-large-uncased-1m_lr1e-5_l12', 'acc': 0.9630627164293959, 'f1': 0.8330503678551217, 'rec': 0.8292957746478873, 'pre': 0.8311688311688312},
# {'task': 'ner-prosa', 'model': 'fasttext-4B-id-300-no-oov-uncased_lr1e-4_l4', 'acc': 0.9346383224317045, 'f1': 0.5396897810218978, 'rec': 0.6664788732394367, 'pre': 0.5964204688681624},
# {'task': 'ner-prosa', 'model': 'babert-bpe-mlm-large-uncased-1100k_lr1e-5_l12', 'acc': 0.9660927279722971, 'f1': 0.8423973362930077, 'rec': 0.8552112676056338, 'pre': 0.8487559407324573},
# {'task': 'ner-prosa', 'model': 'fasttext-cc-id-300-no-oov-uncased_lr1e-4_l4', 'acc': 0.9286263947672182, 'f1': 0.5279027902790279, 'rec': 0.6608450704225353, 'pre': 0.5869402051538654},
# {'task': 'ner-prosa', 'model': 'albert-base-uncased-96000_lr1e-5_l12', 'acc': 0.9513755290496345, 'f1': 0.7722592368261659, 'rec': 0.7183098591549296, 'pre': 0.74430823117338},
# {'task': 'ner-prosa', 'model': 'albert-base-uncased-191k_lr1e-5_l12', 'acc': 0.9506060023085803, 'f1': 0.7706855791962175, 'rec': 0.7346478873239437, 'pre': 0.7522353619844245},
# {'task': 'ner-prosa', 'model': 'albert-base-uncased-112500_lr1e-5_l12', 'acc': 0.9497402847248941, 'f1': 0.7546158427635498, 'rec': 0.7138028169014085, 'pre': 0.7336421540243196},
# {'task': 'pos-idn', 'model': 'fasttext-cc-id-300-no-oov-uncased_lr1e-4_l2', 'acc': 0.9258484509683702, 'f1': 0.9112158485649692, 'rec': 0.9248599875544493, 'pre': 0.9179872222972844},
# {'task': 'pos-idn', 'model': 'bert-base-multilingual-uncased_lr1e-5_l12', 'acc': 0.9603996104879721, 'f1': 0.9511478704490994, 'rec': 0.9571406347230865, 'pre': 0.9541348427868025},
# {'task': 'pos-idn', 'model': 'fasttext-cc-id-300-no-oov-uncased_lr1e-4_l6', 'acc': 0.9270025606809247, 'f1': 0.9150776564662464, 'rec': 0.9257545115121344, 'pre': 0.9203851210269894},
# {'task': 'pos-idn', 'model': 'babert-opensubtitle_lr1e-5_l12', 'acc': 0.9498683593609117, 'f1': 0.9380667567046113, 'rec': 0.9454729309271935, 'pre': 0.9417552830883066},
# {'task': 'pos-idn', 'model': 'babert-base-512_lr1e-5_l12', 'acc': 0.9626356980560464, 'f1': 0.9551915107857946, 'rec': 0.9592408214063473, 'pre': 0.9572118836473716},
# {'task': 'pos-idn', 'model': 'xlm-roberta-large_lr1e-5_l24', 'acc': 0.973960399610488, 'f1': 0.9689624363904751, 'rec': 0.970130678282514, 'pre': 0.9695462054222136},
# {'task': 'pos-idn', 'model': 'fasttext-4B-id-300-no-oov-uncased_lr1e-4_l6', 'acc': 0.9340714826703214, 'f1': 0.918368128710975, 'rec': 0.9324051026757934, 'pre': 0.9253333847964954},
# {'task': 'pos-idn', 'model': 'scratch_lr6.25e-5_l4', 'acc': 0.89519241172864, 'f1': 0.8793937068303914, 'rec': 0.8912958929682638, 'pre': 0.8853047979602875},
# {'task': 'pos-idn', 'model': 'scratch_lr6.25e-5_l6', 'acc': 0.8971760377970931, 'f1': 0.8834025534533149, 'rec': 0.8934349719975109, 'pre': 0.8883904400959084},
# {'task': 'pos-idn', 'model': 'babert-bpe-mlm-uncased-128-dup10-5_lr1e-5_l12', 'acc': 0.9631045551267717, 'f1': 0.9558681080243326, 'rec': 0.9594741754822651, 'pre': 0.9576677471322375},
# {'task': 'pos-idn', 'model': 'scratch_lr6.25e-5_l2', 'acc': 0.8946874887293973, 'f1': 0.878244201648457, 'rec': 0.8909847542003734, 'pre': 0.8845686043593257},
# {'task': 'pos-idn', 'model': 'cartobert_lr1e-5_l12', 'acc': 0.907563025210084, 'f1': 0.8859047619047619, 'rec': 0.9044415059116366, 'pre': 0.8950771717793773},
# {'task': 'pos-idn', 'model': 'fasttext-4B-id-300-no-oov-uncased_lr1e-4_l2', 'acc': 0.9308255491037617, 'f1': 0.9145152036718301, 'rec': 0.9299159925326695, 'pre': 0.9221513006922883},
# {'task': 'pos-idn', 'model': 'xlm-roberta-base_lr1e-5_l12', 'acc': 0.9668914776210914, 'f1': 0.9563942270587328, 'rec': 0.9639079029247044, 'pre': 0.9601363653972804},
# {'task': 'pos-idn', 'model': 'xlm-mlm-100-1280_lr1e-5_l24', 'acc': 0.973311212897176, 'f1': 0.9678997011217638, 'rec': 0.9698195395146235, 'pre': 0.968858669256921},
# {'task': 'pos-idn', 'model': 'babert-bpe-mlm-large-uncased_lr1e-5_l12', 'acc': 0.9652685108378115, 'f1': 0.9576776994031471, 'rec': 0.9610298693217175, 'pre': 0.9593508560779594},
# {'task': 'pos-idn', 'model': 'babert-bpe-mlm-large-uncased-1m_lr1e-5_l12', 'acc': 0.9655931041944675, 'f1': 0.9588972042343635, 'rec': 0.9617688238954574, 'pre': 0.9603308673617988},
# {'task': 'pos-idn', 'model': 'fasttext-4B-id-300-no-oov-uncased_lr1e-4_l4', 'acc': 0.934576405669564, 'f1': 0.9190483487855337, 'rec': 0.9329884878655881, 'pre': 0.9259659551472574},
# {'task': 'pos-idn', 'model': 'babert-bpe-mlm-large-uncased-1100k_lr1e-5_l12', 'acc': 0.9664586864788834, 'f1': 0.959530177927666, 'rec': 0.9627022401991289, 'pre': 0.9611135917994914},
# {'task': 'pos-idn', 'model': 'fasttext-cc-id-300-no-oov-uncased_lr1e-4_l4', 'acc': 0.9279763407508926, 'f1': 0.9165513530135302, 'rec': 0.9273879900435594, 'pre': 0.921937828642128},
# {'task': 'pos-idn', 'model': 'albert-base-uncased-96000_lr1e-5_l12', 'acc': 0.9518880513578822, 'f1': 0.9397641436719593, 'rec': 0.9483898568761667, 'pre': 0.9440572977158342},
# {'task': 'pos-idn', 'model': 'albert-base-uncased-191k_lr1e-5_l12', 'acc': 0.946838821365456, 'f1': 0.9319586837153938, 'rec': 0.9439561294337274, 'pre': 0.937919041638489},
# {'task': 'pos-idn', 'model': 'albert-base-uncased-112500_lr1e-5_l12', 'acc': 0.9527896995708155, 'f1': 0.9401571164510166, 'rec': 0.9495177349097698, 'pre': 0.9448142414860681},
# {'task': 'qa-factoid-itb', 'model': 'fasttext-4B-id-300-no-oov-uncased_lr1e-4_l2', 'acc': 0.9457261724659607, 'f1': 0.3137254901960784, 'rec': 0.050156739811912224, 'pre': 0.08648648648648648},
# {'task': 'qa-factoid-itb', 'model': 'fasttext-cc-id-300-no-oov-uncased_lr1e-4_l6', 'acc': 0.9408093797276853, 'f1': 0.15803108808290156, 'rec': 0.19122257053291536, 'pre': 0.17304964539007092},
# {'task': 'qa-factoid-itb', 'model': 'cartobert_lr1e-5_l12', 'acc': 0.9451588502269289, 'f1': 0.2668463611859838, 'rec': 0.3103448275862069, 'pre': 0.28695652173913044},
# {'task': 'qa-factoid-itb', 'model': 'albert-base-uncased-191k_lr1e-5_l12', 'acc': 0.9585854765506808, 'f1': 0.5579710144927537, 'rec': 0.4827586206896552, 'pre': 0.5176470588235293},
# {'task': 'qa-factoid-itb', 'model': 'babert-bpe-mlm-large-uncased-1m_lr1e-5_l12', 'acc': 0.9765506807866868, 'f1': 0.6189111747851003, 'rec': 0.677115987460815, 'pre': 0.6467065868263473},
# {'task': 'qa-factoid-itb', 'model': 'fasttext-cc-id-300-no-oov-uncased_lr1e-4_l2', 'acc': 0.9384770549672213, 'f1': 0.12314225053078556, 'rec': 0.18181818181818182, 'pre': 0.14683544303797466},
# {'task': 'qa-factoid-itb', 'model': 'albert-base-uncased-96000_lr1e-5_l12', 'acc': 0.9706253151790217, 'f1': 0.4883116883116883, 'rec': 0.5893416927899686, 'pre': 0.5340909090909091},
# {'task': 'qa-factoid-itb', 'model': 'scratch_lr6.25e-5_l4', 'acc': 0.9358295511850732, 'f1': 0.08611111111111111, 'rec': 0.09717868338557993, 'pre': 0.09131075110456555},
# {'task': 'qa-factoid-itb', 'model': 'scratch_lr6.25e-5_l2', 'acc': 0.9375945537065052, 'f1': 0.09202453987730061, 'rec': 0.09404388714733543, 'pre': 0.09302325581395347},
# {'task': 'qa-factoid-itb', 'model': 'bert-base-multilingual-uncased_lr1e-5_l12', 'acc': 0.9730206757438224, 'f1': 0.603448275862069, 'rec': 0.658307210031348, 'pre': 0.6296851574212893},
# {'task': 'qa-factoid-itb', 'model': 'babert-base-512_lr1e-5_l12', 'acc': 0.9661497730711044, 'f1': 0.505524861878453, 'rec': 0.5736677115987461, 'pre': 0.5374449339207049},
# {'task': 'qa-factoid-itb', 'model': 'fasttext-4B-id-300-no-oov-uncased_lr1e-4_l4', 'acc': 0.9471759959657086, 'f1': 0.2711864406779661, 'rec': 0.10031347962382445, 'pre': 0.14645308924485126},
# {'task': 'qa-factoid-itb', 'model': 'babert-opensubtitle_lr1e-5_l12', 'acc': 0.9508320726172466, 'f1': 0.3322475570032573, 'rec': 0.31974921630094044, 'pre': 0.32587859424920124},
# {'task': 'qa-factoid-itb', 'model': 'fasttext-4B-id-300-no-oov-uncased_lr1e-4_l6', 'acc': 0.9476802824004035, 'f1': 0.23645320197044334, 'rec': 0.15047021943573669, 'pre': 0.1839080459770115},
# {'task': 'qa-factoid-itb', 'model': 'xlm-roberta-base_lr1e-5_l12', 'acc': 0.9742183560262229, 'f1': 0.6158536585365854, 'rec': 0.6332288401253918, 'pre': 0.624420401854714},
# {'task': 'qa-factoid-itb', 'model': 'albert-base-uncased-112500_lr1e-5_l12', 'acc': 0.9728315683308119, 'f1': 0.510752688172043, 'rec': 0.5956112852664577, 'pre': 0.5499276410998553},
# {'task': 'qa-factoid-itb', 'model': 'fasttext-cc-id-300-no-oov-uncased_lr1e-4_l4', 'acc': 0.94232223903177, 'f1': 0.1553398058252427, 'rec': 0.15047021943573669, 'pre': 0.15286624203821655},
# {'task': 'qa-factoid-itb', 'model': 'babert-bpe-mlm-large-uncased-1100k_lr1e-5_l12', 'acc': 0.9771180030257186, 'f1': 0.6358381502890174, 'rec': 0.6896551724137931, 'pre': 0.6616541353383458},
# {'task': 'qa-factoid-itb', 'model': 'babert-bpe-mlm-uncased-128-dup10-5_lr1e-5_l12', 'acc': 0.968166918809884, 'f1': 0.4986376021798365, 'rec': 0.5736677115987461, 'pre': 0.5335276967930029},
# {'task': 'qa-factoid-itb', 'model': 'babert-bpe-mlm-large-uncased_lr1e-5_l12', 'acc': 0.9773071104387292, 'f1': 0.5925925925925926, 'rec': 0.6520376175548589, 'pre': 0.6208955223880597},
# {'task': 'qa-factoid-itb', 'model': 'scratch_lr6.25e-5_l6', 'acc': 0.9444024205748865, 'f1': 0.14814814814814814, 'rec': 0.012539184952978056, 'pre': 0.023121387283236993},
# {'task': 'qa-factoid-itb', 'model': 'xlm-roberta-large_lr1e-5_l24', 'acc': 0.9838628340897629, 'f1': 0.7242424242424242, 'rec': 0.7492163009404389, 'pre': 0.7365177195685672},
# {'task': 'qa-factoid-itb', 'model': 'xlm-mlm-100-1280_lr1e-5_l24', 'acc': 0.978315683308119, 'f1': 0.5872576177285319, 'rec': 0.664576802507837, 'pre': 0.6235294117647059},
# {'task': 'absa-airy', 'model': 'xlm-mlm-100-1280_lr1e-5_l24', 'acc': 0.9171328671328671, 'f1': 0.7941056388294077, 'rec': 0.7632583265461775, 'pre': 0.8323715095971885},
# {'task': 'absa-airy', 'model': 'albert-base-wwmlm-512_lr1e-5_l12', 'acc': 0.9527972027972028, 'f1': 0.8803746215235321, 'rec': 0.8462506664483054, 'pre': 0.9235987145941905}
# {'task': 'absa-airy', 'model': 'albert-large-wwmlm-128_lr1e-5_l12', 'acc': 0.956993006993007, 'f1': 0.8952814307733683, 'rec': 0.8763083046885826, 'pre': 0.91675232655831}
# {'task': 'absa-airy', 'model': 'babert-bpe-mlm-large-512_lr1e-5_l24', 'acc': 0.9667832167832168, 'f1': 0.9231033339118567, 'rec': 0.9147976398800594, 'pre': 0.9318257765626187}
# {'task': 'pos-prosa', 'model': 'fasttext-4B-id-300-no-oov-uncased_lr1e-4_l6', 'acc': 0.9611869949980761, 'f1': 0.9595529270248596, 'rec': 0.9595529270248596, 'pre': 0.9595529270248596}
# ]
if args["all"] or args["task"] == "":
    for task in os.listdir(f'save/'):
        for dir_name in os.listdir(f'save/{task}'):
            model_name, bs, _, _, lr, early, layer, *_ = dir_name.split("_")
            acc, f1, rec, pre = None, None, None, None

            path = f'save/{task}/{dir_name}/evaluation_result.csv' 
            if os.path.exists(path):
                with open(path) as f:
                    for line in f:
                        splits = line.replace("\n","").split(",")
                        if splits[0] == "mean":
                            acc, f1, rec, pre = splits[1:]
                            break
                rows.append({"task": task, "model":model_name + "_" + lr + "_" + layer.replace("layer","l"), "acc":float(acc), "f1":float(f1), "rec":float(rec), "pre":float(pre)})
                # print({"task": task, "model":model_name + "_" + lr + "_" + layer.replace("layer","l"), "acc":float(acc), "f1":float(f1), "rec":float(rec), "pre":float(pre)})
    
    task_map_to_alias = {

    }
    # print(rows)sss
    rows = pd.DataFrame(rows)
    rows = rows.pivot(index="model", columns='task', values=['f1'])
    rows['average'] = rows.mean(numeric_only=True, axis=1)
    rows = rows.sort_values(by = 'average', ascending=False)
    rows = rows.rename(lambda x: x[:8], axis=1)
    print(rows)
    # print(rows.mean(axis=1))
    # print(tabulate(rows,headers="keys",tablefmt='simple',floatfmt=".2f",numalign="center"))
else:
    task = args["task"]
    for dir_name in os.listdir(f'save/{task}'):
        model_name, bs, _, _, lr, early, layer, *_ = dir_name.split("_")
        acc, f1, rec, pre = None, None, None, None

        path = f'save/{task}/{dir_name}/evaluation_result.csv' 
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    splits = line.split(",")
                    if splits[0] == "mean":
                        acc, f1, rec, pre = splits[1:]
                        break
            rows.append({"model":model_name, "lr":lr.replace("lr",""), "bs":bs.replace("b",""), "early":early.replace("early",""), "acc":acc, "f1":f1, "rec":rec, "pre":pre, "layer":layer})
            print({"model":model_name, "lr":lr.replace("lr",""), "bs":bs.replace("b",""), "early":early.replace("early",""), "acc":acc, "f1":f1, "rec":rec, "pre":pre, "layer":layer})

    rows = sorted(rows, key=lambda i:i['f1'])
    print(tabulate(rows,headers="keys",tablefmt='latex',floatfmt=".8f",numalign="center"))
