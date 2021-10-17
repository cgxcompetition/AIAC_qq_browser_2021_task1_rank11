import json
from pathlib import Path

import numpy as np
from sklearn import decomposition


def l2(x):
    return x / np.linalg.norm(x, axis=0, keepdims=True)


js_dict = {}
for fold in Path('./save/output/').glob('fold[0-9]*_testb.json'):
    with open(fold) as f:
        js_dict[fold.stem] = json.load(f)
keys = list(js_dict.keys())
concat_result = {}
for key, value in js_dict.get(list(js_dict.keys())[0]).items():
    concat_result[key] = (
        np.concatenate(
            [
                l2(np.array(value)),
                l2(np.array(js_dict[keys[1]][key])),
                l2(np.array(js_dict[keys[2]][key])),
                l2(np.array(js_dict[keys[3]][key])),
                l2(np.array(js_dict[keys[4]][key])),
            ],
            axis=0
        )
    ).tolist()


concat_svd_result = {}
result_key = []
result_value = []

for key, value in concat_result.items():
    result_key.append(key)
    result_value.append(value)
estimator = decomposition.TruncatedSVD(
    n_components=256,
    algorithm='arpack'
)
svd_X_train = estimator.fit_transform(np.array(result_value))
for i in range(svd_X_train.shape[0]):
    concat_svd_result[result_key[i]] = svd_X_train[i].tolist()

with open('../output/gpj_result.json', 'w') as f:
    json.dump(concat_svd_result, f)
