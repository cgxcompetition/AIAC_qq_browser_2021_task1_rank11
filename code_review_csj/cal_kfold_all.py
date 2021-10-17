import json
from zipfile import ZIP_DEFLATED, ZipFile
import os
import numpy as np
from config import parser
from sklearn import decomposition

def l2(x):
    y = np.linalg.norm(x, axis=0, keepdims=True)
    z = x/y
    return z

def main():
    parser.add_argument('--use_loss', default='mse', type=str, help='kfold')
    args = parser.parse_args()
    result = []
    for i in range(5):
        savedmodel_path = args.savedmodel_path.format(fold=i, loss=args.use_loss)

        result_json = os.path.join(savedmodel_path, args.output_json)
        print(result_json)
        with open(result_json) as f:
            vid_embedding = json.load(f)
        result.append(vid_embedding)
    vid1, vid2, vid3, vid4, vid5 = result

    new1 = {}
    result_value = []
    keys = []
    for key in vid1:
        x1 = l2(np.array(vid1[key]))
        x2 = l2(np.array(vid2[key]))
        x3 = l2(np.array(vid3[key]))
        x4 = l2(np.array(vid4[key]))
        x5 = l2(np.array(vid5[key]))

        s1 = np.concatenate([x1, x2,x3,x4,x5], axis=0)
        new1[key] = s1.tolist()
        result_value.append(s1.tolist())
        keys.append(key)
    estimator = decomposition.TruncatedSVD(
        n_components=256,
        algorithm='arpack')
    concat_svd_result = {}
    svd_X_train = estimator.fit_transform(np.array(result_value))
    for i in range(svd_X_train.shape[0]):
        concat_svd_result[keys[i]] = svd_X_train[i].tolist()
    base_path = os.path.dirname(os.getcwd())
    result_path = os.path.join(base_path, 'output/csj_result_{loss}.json')
    with open(result_path.format(loss=args.use_loss), 'w') as f:
        json.dump(concat_svd_result, f)


if __name__ == '__main__':
    main()
