import argparse

parser = argparse.ArgumentParser(description="QQ Browser video embedding challenge")

parser.add_argument('--dropout', type=float, default=0.2, help='dropout ratio')
parser.add_argument('--multi-label-file', type=str, default='../data/tag_list.txt', help='supervised tag list')

# ========================= Dataset Configs ==========================
parser.add_argument('--train-record-pattern', type=str, default='../save/data/pairwise_train_fold_0.tfrecords')
parser.add_argument('--pretrain-record-pattern', type=str, default='../save/data/train_pointwise.tfrecords')
parser.add_argument('--val-record-pattern', type=str, default='../save/data/pairwise_val_fold_0.tfrecords')
parser.add_argument('--annotation-file', type=str, default='../data/pairwise/label.tsv')
parser.add_argument('--test-a-file', type=str, default='../save/data/test_a.tfrecords')
parser.add_argument('--test-b-file', type=str, default='../save/data/test_b.tfrecords')
parser.add_argument('--output-json', type=str, default='result_fold1.json')
parser.add_argument('--output-zip', type=str, default='result_asr_text_fold1_tag12000_testb.zip')
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--val-batch-size', default=64, type=int)
parser.add_argument('--test-batch-size', default=64, type=int)

# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', default=50, type=int, help='print frequency')
parser.add_argument('--eval-freq', default=1000, type=int, help='evaluation step frequency')

# ======================== SavedModel Configs =========================
parser.add_argument('--resume-training', default=1, type=int, help='resume training from checkpoints')
parser.add_argument('--savedmodel-path', type=str, default='../save/ft_fold1/20211001')
parser.add_argument('--ckpt-file', type=str, default='../save/ft_fold1/20211001/ckpt-150000')
parser.add_argument('--pretrain-savedmodel-path', type=str, default='../save/pretrain/20211001')
parser.add_argument('--pretrain-ckpt-file', type=str, default='../save/pretrain/20211001/ckpt-60000')
parser.add_argument('--max-to-keep', default=20, type=int, help='the number of checkpoints to keep')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')

# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=120, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--total-steps', default=150000, type=int)
parser.add_argument('--warmup-steps', default=2000, type=int)
parser.add_argument('--minimum-lr', default=0., type=float, help='minimum learning rate')
parser.add_argument('--lr', default=0.0005, type=float, help='initial learning rate')

# ==================== Vision Modal Configs =======================
parser.add_argument('--frame-embedding-size', type=int, default=1536)
parser.add_argument('--max-frames', type=int, default=32)
parser.add_argument('--vlad-cluster-size', type=int, default=64)
parser.add_argument('--vlad-groups', type=int, default=8)
parser.add_argument('--vlad-hidden-size', type=int, default=1024, help='nextvlad output size using dense')
parser.add_argument('--se-ratio', type=int, default=8, help='reduction factor in se context gating')

# ========================== Title BERT =============================
parser.add_argument('--bert-dir', type=str, default='../data/chinese_L-12_H-768_A-12')
parser.add_argument('--bert-seq-length', type=int, default=32)
parser.add_argument('--bert-lr', type=float, default=3e-5)
parser.add_argument('--bert-total-steps', type=int, default=150000)
parser.add_argument('--bert-warmup-steps', type=int, default=2000)

# ====================== Fusion Configs ===========================
parser.add_argument('--hidden-size', type=int, default=256, help='NO MORE THAN 256')
